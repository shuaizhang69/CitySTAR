#!/usr/bin/env python3
"""
Convert stage1 candidates jsonl (with 'candidates' field) to evaluation log format
(with candidates_30, candidates_20, candidates_10, is_success)
"""
import json
import argparse
from pathlib import Path


def convert_candidates(input_jsonl: str, output_json: str):
    """Convert candidates format to evaluation log format."""
    results = []
    
    with open(input_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            item = json.loads(line)
            candidates = item.get('candidates', [])
            
            # Create evaluation log format
            result = {
                'scene_id': item['scene_id'],
                'object_id': item['object_id'],
                'ann_id': item.get('ann_id', 0),
                'is_success': len(candidates) > 0,
                'candidates_30': candidates[:30] if len(candidates) >= 30 else candidates,
                'candidates_20': candidates[:20] if len(candidates) >= 20 else candidates,
                'candidates_10': candidates[:10] if len(candidates) >= 10 else candidates,
            }
            results.append(result)
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Converted {len(results)} rows")
    print(f"Output saved to: {output_json}")
    
    # Stats
    has_30 = sum(1 for r in results if len(r['candidates_30']) >= 30)
    has_20 = sum(1 for r in results if len(r['candidates_20']) >= 20)
    has_10 = sum(1 for r in results if len(r['candidates_10']) >= 10)
    print(f"Rows with >=30 candidates: {has_30}")
    print(f"Rows with >=20 candidates: {has_20}")
    print(f"Rows with >=10 candidates: {has_10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input candidates jsonl')
    parser.add_argument('--output', required=True, help='Output evaluation log json')
    args = parser.parse_args()
    
    convert_candidates(args.input, args.output)
