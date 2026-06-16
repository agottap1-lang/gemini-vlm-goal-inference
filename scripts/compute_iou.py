#!/usr/bin/env python3
"""Compute IoU and time-to-legibility metrics between VLM and human annotations."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_vlm_eval.schema import EvaluationResult

def load_jsonl(file_path: str) -> Dict[str, List[EvaluationResult]]:
    """Load JSONL file and group by video."""
    data = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                result = EvaluationResult.model_validate_json(line)
                data[result.video].append(result)
    return data

def compute_metrics(vlm_data: Dict[str, List[EvaluationResult]],
                   human_data: Dict[str, List[EvaluationResult]]) -> List[Dict]:
    """Compute IoU and time-to-legibility for each video."""
    results = []
    for video in set(vlm_data.keys()) | set(human_data.keys()):
        vlm_results = vlm_data.get(video, [])
        human_results = human_data.get(video, [])

        # Get sets of t_sec where legible_now
        vlm_legible = {r.t_sec for r in vlm_results if r.legible == "legible_now"}
        human_legible = {r.t_sec for r in human_results if r.legible == "legible_now"}

        # IoU
        intersection = vlm_legible & human_legible
        union = vlm_legible | human_legible
        iou = len(intersection) / len(union) if union else 0.0

        # Time-to-legibility
        vlm_ttl = min((r.t_sec for r in vlm_results if r.legible == "legible_now"), default=float('inf'))
        human_ttl = min((r.t_sec for r in human_results if r.legible == "legible_now"), default=float('inf'))

        results.append({
            'video': video,
            'iou': iou,
            'vlm_time_to_legibility': vlm_ttl if vlm_ttl != float('inf') else None,
            'human_time_to_legibility': human_ttl if human_ttl != float('inf') else None,
            'vlm_legible_count': len(vlm_legible),
            'human_legible_count': len(human_legible),
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="Compute IoU and time-to-legibility metrics")
    parser.add_argument("--vlm-jsonl", required=True, help="VLM evaluation JSONL file")
    parser.add_argument("--human-jsonl", required=True, help="Human annotation JSONL file")
    parser.add_argument("--output-csv", help="Output CSV file (optional, prints table if not specified)")

    args = parser.parse_args()

    # Load data
    vlm_data = load_jsonl(args.vlm_jsonl)
    human_data = load_jsonl(args.human_jsonl)

    # Compute metrics
    metrics = compute_metrics(vlm_data, human_data)

    # Output
    if args.output_csv:
        import csv
        with open(args.output_csv, 'w', newline='') as f:
            if metrics:
                writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
                writer.writeheader()
                writer.writerows(metrics)
        print(f"Metrics saved to {args.output_csv}")
    else:
        # Print table
        if metrics:
            headers = list(metrics[0].keys())
            print("\t".join(headers))
            for row in metrics:
                print("\t".join(str(row[h]) for h in headers))

if __name__ == "__main__":
    main()