# Raw Data Placeholder

Place non-committed raw data here. See `docs/DATA_LAYOUT.md` for the expected
directory structure.

This directory is intentionally empty in the code bundle.

For CityAnchor, instance-segmentation outputs can be placed under:

```text
data/raw/cityanchor/instances/
```

Then generate bbox JSON with:

```bash
python scripts/instances_to_bbox.py \
  --input-root data/raw/cityanchor/instances \
  --output-dir data/raw/cityanchor/bbox \
  --recursive
```
