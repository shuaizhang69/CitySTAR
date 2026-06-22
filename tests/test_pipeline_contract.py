from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from city_pipeline.config import available_configs, load_pipeline, render_context, stage_names
from city_pipeline.instance_bbox import convert_instances_to_bbox, parse_xyz_cols
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

    def test_cityanchor_has_instance_to_bbox_stage(self) -> None:
        pipeline = load_pipeline(dataset="cityanchor")
        self.assertEqual(stage_names(pipeline)[0], "bbox_from_instances")

    def test_instance_txt_to_bbox_json(self) -> None:
        outputs = Path.cwd() / "outputs"
        outputs.mkdir(exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="instance_bbox_", dir=str(outputs)))
        try:
            src = tmp / "scene_a_building.txt"
            src.write_text(
                "\n".join(
                    [
                        "0 0 0 10",
                        "2 4 6 10",
                        "10 10 0 12",
                        "14 12 2 12",
                    ]
                ),
                encoding="utf-8",
            )
            out_dir = tmp / "bbox"
            written = convert_instances_to_bbox(
                input_roots=[tmp],
                output_dir=out_dir,
                recursive=False,
                xyz_cols=parse_xyz_cols("0,1,2"),
                instance_col=-1,
                semantic_col=None,
                delimiter=None,
                category=None,
                default_object_name="Object",
                min_points=1,
                object_id_mode="preserve",
                duplicate_id_policy="error",
            )
            self.assertEqual(len(written), 1)
            data = json.loads(written[0].read_text(encoding="utf-8"))
            self.assertEqual(data["scene_id"], "scene_a")
            self.assertEqual([item["object_id"] for item in data["bboxes"]], [10, 12])
            first = data["bboxes"][0]
            self.assertEqual(first["object_name"], "Building")
            self.assertEqual(first["bbox"][:6], [1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

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
