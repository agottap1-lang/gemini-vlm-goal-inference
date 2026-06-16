#!/usr/bin/env python3
"""
DEPRECATED: Use eval_dataset.py instead for manifest-driven evaluation.

This script evaluates a single video without manifest metadata.
For research-grade evaluation with full provenance, use:
  python scripts/eval_dataset.py --manifest data/manifest.jsonl --k all --out outputs/results.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.runner import evaluate_video

def main():
    parser = argparse.ArgumentParser(description="Evaluate robot motion legibility in videos using Gemini VLM")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Sample rate in seconds (default: 1.0 for one frame per second)")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process")
    parser.add_argument("--out", help="Output JSONL file path (optional, auto-generated from video name if not provided)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        client = GeminiClient(model=args.model)
        output_file = evaluate_video(args.video_path, client, args.out, args.sample_rate, args.max_frames)
        print(f"Evaluation complete. Results saved to {output_file}")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()