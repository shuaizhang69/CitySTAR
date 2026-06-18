from __future__ import annotations

import unittest

from city_pipeline.config import available_configs, load_pipeline, render_context, stage_names
from city_pipeline.runner import command_plan, doctor, select_stages


class PipelineContractTest(unittest.TestCase):
    def test_all_configs_load_and_have_unique_stages(self) -> None:
        configs = available_configs()
        self.assertGreaterEqual(len(configs), 3)

        for path in configs:
            pipeline = load_pipeline(config_path=str(path))
            names = stage_names(pipeline)
            self.assertTrue(names, msg=str(path))
            self.assertEqual(len(names), len(set(names)), msg=str(path))

    def test_cityrefer_bundled_replay_is_self_contained(self) -> None:
        pipeline = load_pipeline(dataset="cityrefer")
        report = doctor(pipeline, split="ND", names=["bundle_replay_hgmatch"])
        self.assertTrue(report["ok"], msg=report)

    def test_cityanchor_v7_stage_is_wired_with_overrides(self) -> None:
        pipeline = load_pipeline(dataset="cityanchor")
        plan = command_plan(pipeline, split="ND", names=["hypergraph_match_v7"])
        self.assertEqual(len(plan), 1)
        command = plan[0]["command"]
        for flag in (
            "--override-candidates",
            "--override-desc",
            "--override-bbox",
            "--override-geometry",
            "--override-bbox-dir-geometry",
            "--override-output",
        ):
            self.assertIn(flag, command)

    def test_sam3_pipeline_has_expected_end_to_end_stages(self) -> None:
        pipeline = load_pipeline(dataset="sam3_sensaturban")
        self.assertEqual(
            stage_names(pipeline),
            [
                "grid_split",
                "superpoints",
                "render_views",
                "sam3_masks",
                "semantic_fusion",
                "merge_instances",
                "bbox_json",
            ],
        )

    def test_unknown_stage_fails_fast(self) -> None:
        pipeline = load_pipeline(dataset="cityrefer")
        with self.assertRaises(ValueError):
            select_stages(pipeline, names=["missing_stage"])

    def test_render_context_maps_cityanchor_split_dirs(self) -> None:
        self.assertEqual(render_context("ND")["cityanchor_split_dir"], "CityAnchor")
        self.assertEqual(render_context("NO")["cityanchor_split_dir"], "city_Anchor")


if __name__ == "__main__":
    unittest.main()
