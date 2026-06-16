#!/usr/bin/env python3
"""Evaluate specific timestamps for a single video using the manifest."""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.schema import ManifestEntry


def find_entry(entries, video_id: Optional[str], video_path: Optional[str]):
    if video_id:
        for e in entries:
            if e.video_id == video_id:
                return e
        raise SystemExit(f"video_id '{video_id}' not found in manifest")
    if video_path:
        vp_norm = Path(video_path).as_posix()
        for e in entries:
            if Path(e.video_path).as_posix().endswith(Path(vp_norm).as_posix()):
                return e
        raise SystemExit(f"video_path '{video_path}' not found in manifest")
    raise SystemExit("Provide either --video-id or --video-path")


def load_manifest(manifest_path: str) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(ManifestEntry(**data))
    return entries


def extract_frame_at_time(video_path: str, t_sec: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = int(t_sec * fps)
        if frame_num >= total_frames:
            raise ValueError(f"Requested t={t_sec}s is beyond video length")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame at t={t_sec}s (frame {frame_num})")
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            raise ValueError("Failed to encode frame to JPEG")
        return frame_num, buf.tobytes()
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description="Evaluate specific timestamps for one video")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--video-id", help="video_id from manifest")
    parser.add_argument("--video-path", help="Path to video if not using video_id")
    parser.add_argument("--times", required=True, help="Comma-separated seconds, e.g. 1,3,5,7,9,11")
    parser.add_argument("--out", required=True, help="Output JSONL file path")
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    entries = load_manifest(args.manifest)
    entry = find_entry(entries, args.video_id, args.video_path)

    times: List[int] = []
    for tok in args.times.split(','):
        tok = tok.strip()
        if not tok:
            continue
        times.append(int(tok))
    times = sorted(set(times))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = GeminiClient(model=args.model)
    with open(out_path, 'w') as f:
        for t in times:
            try:
                frame_idx, img_bytes = extract_frame_at_time(entry.video_path, t)
                result = client.evaluate_frame(img_bytes, entry, t, frame_idx)
                f.write(result.model_dump_json() + "\n")
                f.flush()
            except Exception as e:
                logging.error(f"Failed at t={t}s: {e}")

    print(f"Done. Wrote {out_path}")


if __name__ == "__main__":
    main()
