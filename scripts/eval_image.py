#!/usr/bin/env python3
"""CLI script for evaluating images with Gemini VLM."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.runner import evaluate_image

def main():
    parser = argparse.ArgumentParser(description="Evaluate robot motion legibility in images using Gemini VLM")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model to use")
    parser.add_argument("--out", default="outputs/image_results.jsonl", help="Output JSONL file path")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        client = GeminiClient(model=args.model)
        evaluate_image(args.image_path, client, args.out)
        print(f"Evaluation complete. Results saved to {args.out}")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()