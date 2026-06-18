# CityRefer Pipeline

Config: `configs/cityrefer.pipeline.json`

## Inputs

Bundled inputs:

```text
cityrefer/data/Cityrefer/meta_data/0311data/CityRefer_val_{ND,NO}_0324.jsonl
cityrefer/data/Cityrefer/meta_data/0311data/CityRefer_val_{ND,NO}_0421_stage1_candidates.jsonl
cityrefer/data/Cityrefer/0311data/feature/all_objects_color_per_image_tier2.jsonl
cityrefer/data/Cityrefer/0412/desc/CityRefer_val_{ND,NO}_desc_hypergraphs.jsonl
cityrefer/data/Cityrefer/0412/desc_bbox_hypergraphs/*_bbox_only_hypergraphs.jsonl
cityrefer/data/Cityrefer/0412/hypergraph_match/*_hgmatch_v2.json
```

External inputs for from-scratch runs:

```text
data/raw/cityrefer/bbox/
data/raw/cityrefer/desc/CityRefer_val_{ND,NO}_infer_result.jsonl
data/raw/cityrefer/context_images/
data/raw/sensaturban/instances/
```

## Stages

1. `stage1_candidates`

   Runs `cityrefer/pipeline2/01_Semantic_landmark_in_stage1_new.py`.
   It uses the metadata JSONL, bbox JSON, and object color tiers to produce a
   candidate list per query.

2. `stage1_eval_log`

   Converts stage1 candidates into the remote matcher format:

   ```json
   {
     "scene_id": "...",
     "object_id": "...",
     "ann_id": 0,
     "is_success": true,
     "candidates_30": ["..."],
     "candidates_20": ["..."],
     "candidates_10": ["..."]
   }
   ```

3. `desc_hypergraphs`

   Runs `cityrefer/superGraph/2_generate_desc_hypergraphs.py` over LLM NER
   output. This creates query-side nodes and spatial edges.

4. `bbox_hypergraphs`

   Runs `external/sam3_segment/core/generate_bbox_hypergraphs_v2.py`.
   It expands each candidate bbox neighborhood, includes overlapping SAM3
   instances, and builds candidate-side spatial hypergraphs.

5. `hypergraph_match`

   Runs `external/sam3_segment/core/match_hypergraphs.py` on the full output
   directory.

6. `bundle_replay_hgmatch`

   Replays the JSONL artifacts included in this bundle with
   `cityrefer/superGraph/04_hypergraph_match.py`. This is the quickest local
   sanity check because it does not need raw bbox/SAM3 data.

7. `dashscope_rerank`

   Optional. Uses `cityrefer/pipeline2/05reranking_Reranking_fused.py` and
   `DASHSCOPE_API_KEY`.

## Example

```bash
python scripts/run_pipeline.py doctor --dataset cityrefer --split ND
python scripts/run_pipeline.py plan --dataset cityrefer --split ND --stage bundle_replay_hgmatch
python scripts/run_pipeline.py run --dataset cityrefer --split ND --stage bundle_replay_hgmatch
```
