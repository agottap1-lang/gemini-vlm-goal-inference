#!/usr/bin/env python3
"""
DEPRECATED: Use eval_dataset.py with a manifest instead.

For batch evaluation with proper metadata, use:
  python scripts/eval_dataset.py --manifest data/manifest.jsonl --k all --out outputs/results.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.runner import evaluate_folder

def main():
    parser = argparse.ArgumentParser(description="Evaluate all videos in a folder using Gemini VLM")
    parser.add_argument("folder_path", help="Path to the folder containing videos")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Sample rate in seconds")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process per video")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        client = GeminiClient(model=args.model)

        folder_path = Path(args.folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]

        if not video_files:
            logging.warning(f"No video files found in {folder_path}")
            return

        evaluate_folder(args.folder_path, client, args.sample_rate, args.max_frames)

        print(f"Evaluation complete. Results saved to outputs/ directory")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()