#!/usr/bin/env python3
"""
CANONICAL EVALUATION PIPELINE: Evaluate legibility for an entire dataset using manifest.jsonl.

This is the main entrypoint for research-grade, reproducible VLM legibility evaluation.
Produces JSONL output with full API metadata for each frame evaluation.
"""

import argparse
import json
import logging
import sys
import uuid
import subprocess
import platform
import os
from pathlib import Path
from typing import List
from datetime import datetime
from importlib import metadata as importlib_metadata

import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.openai_client import OpenAIClient
from gemini_vlm_eval.anthropic_client import AnthropicClient
from gemini_vlm_eval.schema import ManifestEntry
from gemini_vlm_eval.video import extract_frames, extract_and_cache_frames


PROVIDER_DEFAULTS = {
    "google": "gemini-2.5-flash",
    "openai": "gpt-4o",
    "anthropic": "claude-opus-4-5",
}


def build_client(provider: str, model: str):
    """Instantiate the correct VLM client for the given provider."""
    if provider == "google":
        return GeminiClient(model=model)
    elif provider == "openai":
        return OpenAIClient(model=model)
    elif provider == "anthropic":
        return AnthropicClient(model=model)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: google, openai, anthropic")

def load_manifest(manifest_path: str) -> List[ManifestEntry]:
    """Load manifest.jsonl and return list of entries."""
    entries = []
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                entries.append(ManifestEntry(**data))
    return entries


def get_git_info(repo_root: Path) -> tuple:
    """Return (commit_hash, dirty_flag)."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, bool(status)
    except Exception:
        return "unknown", False


def write_pip_freeze(path: Path) -> None:
    """Write pip freeze to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        path.write_text(result.stdout, encoding="utf-8")
    except Exception as exc:
        path.write_text(f"pip freeze failed: {exc}\n", encoding="utf-8")


def get_optional_machine_info() -> dict:
    """Collect lightweight machine info without extra deps."""
    info = {
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    # Try to get RAM if psutil is available
    try:
        import psutil
        info["ram_bytes"] = psutil.virtual_memory().total
    except Exception:
        info["ram_bytes"] = None
    return info

def evaluate_video_for_k_seconds(
    client, 
    manifest_entry: ManifestEntry, 
    k: int, 
    output_file, 
    mode: str = "single_frame",
    save_frames: bool = False
) -> None:
    """
    Evaluate a single video for first k seconds.
    
    Args:
        client: GeminiClient instance
        manifest_entry: Video metadata
        k: Number of seconds to evaluate (None for all)
        output_file: File handle to write JSONL results
        mode: Evaluation mode - "single_frame" or "prefix_frames"
        save_frames: If True, cache frames to outputs/frames/{video_id}/t_{t_sec:03d}.png
    """
    video_path = manifest_entry.video_path
    if not Path(video_path).exists():
        logging.error(f"Video not found: {video_path}")
        return

    # Extract and cache frames
    frames_dict = extract_and_cache_frames(
        video_path=video_path,
        video_id=manifest_entry.video_id,
        sample_rate_seconds=1.0,
        max_frames=k,
        save_frames=save_frames
    )
    
    # Sort timestamps
    timestamps = sorted(frames_dict.keys())
    
    for t_sec in timestamps:
        try:
            # Prepare image data based on mode
            if mode == "prefix_frames":
                # Send all frames from 0 to t_sec (inclusive)
                prefix_frames = [frames_dict[t]["jpeg_bytes"] for t in range(0, t_sec + 1) if t in frames_dict]
                image_data = prefix_frames
            else:  # single_frame
                image_data = frames_dict[t_sec]["jpeg_bytes"]
            
            # Evaluate
            result = client.evaluate_frame(
                image_bytes=image_data,
                manifest_entry=manifest_entry,
                t_sec=t_sec,
                frame_idx=frames_dict[t_sec]["frame_idx"],
                mode=mode
            )
            output_file.write(result.model_dump_json() + '\n')
            output_file.flush()
        except Exception as e:
            logging.error(f"Failed to evaluate {manifest_entry.video_id} at t={t_sec}s: {e}")
            # Continue with other frames, don't skip the video

def preflight_check(provider: str, model: str) -> bool:
    """
    Validate API key and model access before starting a long evaluation run.
    Generates a valid 32x32 JPEG with cv2 and sends one test call to the API.
    Returns True on success, raises SystemExit on failure.
    """
    import cv2 as _cv2
    import numpy as _np

    print(f"\n{'='*60}")
    print(f"  PRE-FLIGHT CHECK: provider={provider}, model={model}")
    print(f"{'='*60}")

    # Generate a valid 32x32 grey JPEG using cv2
    img = _np.full((32, 32, 3), 128, dtype=_np.uint8)
    ok, buf = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("cv2 failed to encode test image")
    TINY_JPEG = buf.tobytes()

    # Build a dummy ManifestEntry for the test call
    from gemini_vlm_eval.schema import ManifestEntry
    dummy_entry = ManifestEntry(
        video_id="preflight_test",
        video_path="preflight_test.mp4",
        goal_gt="A",
        goal_A="pick the left block",
        goal_B="pick the right block",
        scene_id="test",
        task_family="block_pick",
        traj_type="legible",
        notes="preflight validation call",
    )

    try:
        client = build_client(provider, model)
        result = client.evaluate_frame(
            image_bytes=TINY_JPEG,
            manifest_entry=dummy_entry,
            t_sec=0,
            frame_idx=0,
            mode="single_frame",
        )
        print(f"  ✅  API key valid | model reachable | latency={result.latency_ms}ms")
        print(f"  ✅  Test response: pA={result.pA:.2f}, pB={result.pB:.2f}, cue='{result.cue[:60]}'")
        print(f"{'='*60}\n")
        return True
    except Exception as exc:
        print(f"  ❌  Pre-flight FAILED: {exc}")
        print(f"{'='*60}\n")
        logging.error(f"Pre-flight check failed for {provider}/{model}: {exc}")
        raise SystemExit(1)


def main():
    run_id = str(uuid.uuid4())
    timestamp_utc_start = datetime.utcnow().isoformat() + "Z"
    cli_command = " ".join(sys.argv)
    repo_root = Path(__file__).parent.parent
    
    parser = argparse.ArgumentParser(description="Evaluate legibility for dataset")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--provider", default="google",
                       choices=["google", "openai", "anthropic"],
                       help="VLM provider (default: google)")
    parser.add_argument("--model", default=None,
                       help="Model name override. Defaults: google=gemini-2.5-flash, openai=gpt-4o, anthropic=claude-opus-4-5")
    parser.add_argument("--k", required=True, help="Number of seconds to evaluate (int or 'all')")
    parser.add_argument("--out", required=True, help="Output JSONL file path")
    parser.add_argument("--mode", choices=["single_frame", "prefix_frames"], default="single_frame",
                       help="Evaluation mode: single_frame (baseline) or prefix_frames (temporal context)")
    parser.add_argument("--save-frames", action="store_true",
                       help="Save sampled frames to outputs/frames/{video_id}/t_{t_sec:03d}.png")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse k
    if args.k.lower() == 'all':
        k = None  # No limit
    else:
        k = int(args.k)

    try:
        # Load manifest
        manifest_entries = load_manifest(args.manifest)
        logging.info(f"Loaded {len(manifest_entries)} videos from manifest")

        # Resolve model name (use explicit override or provider default)
        model_name = args.model or PROVIDER_DEFAULTS[args.provider]

        # Initialize client
        logging.info(f"Using provider={args.provider}, model={model_name}")
        client = build_client(args.provider, model_name)

        # Pre-flight: validate key + quota + model BEFORE touching the dataset
        preflight_check(args.provider, model_name)

        # Prepare output
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for entry in manifest_entries:
                logging.info(f"Evaluating {entry.video_id} (mode={args.mode})")
                evaluate_video_for_k_seconds(
                    client=client,
                    manifest_entry=entry,
                    k=k,
                    output_file=f,
                    mode=args.mode,
                    save_frames=args.save_frames
                )

        logging.info(f"Dataset evaluation complete. Results saved to {args.out}")
        
        # Persist run info
        timestamp_utc_end = datetime.utcnow().isoformat() + "Z"
        git_commit, repo_dirty = get_git_info(repo_root)
        python_version = sys.version.replace("\n", " ")
        os_name = platform.system()
        platform_str = platform.platform()
        opencv_version = getattr(cv2, "__version__", "unknown")
        try:
            google_genai_version = importlib_metadata.version("google-genai")
        except Exception:
            google_genai_version = "unknown"

        try:
            openai_version = importlib_metadata.version("openai")
        except Exception:
            openai_version = "unknown"

        try:
            anthropic_version = importlib_metadata.version("anthropic")
        except Exception:
            anthropic_version = "unknown"

        api_key_env_map = {
            "google": "env:GEMINI_API_KEY",
            "openai": "env:OPENAI_API_KEY",
            "anthropic": "env:ANTHROPIC_API_KEY",
        }

        run_info = {
            "run_id": run_id,
            "timestamp_utc_start": timestamp_utc_start,
            "timestamp_utc_end": timestamp_utc_end,
            "git_commit": git_commit,
            "repo_dirty": repo_dirty,
            "python_version": python_version,
            "os": os_name,
            "platform": platform_str,
            "opencv_version": opencv_version,
            "google_genai_version": google_genai_version,
            "openai_version": openai_version,
            "anthropic_version": anthropic_version,
            "provider": args.provider,
            "model": model_name,
            "evaluation_mode": args.mode,
            "save_frames": args.save_frames,
            "command_line": cli_command,
            "manifest_path": args.manifest,
            "output_path": args.out,
            "k_seconds": args.k,
            "num_videos": len(manifest_entries),
            "api_key_source": api_key_env_map[args.provider],
        }

        # pip freeze capture
        pip_freeze_path = output_path.parent / "pip_freeze.txt"
        write_pip_freeze(pip_freeze_path)
        run_info["pip_freeze_path"] = str(pip_freeze_path)

        # Optional machine info
        run_info["machine_info"] = get_optional_machine_info()

        # Save run_info next to outputs
        run_info_path = output_path.parent / f"run_info_{run_id[:8]}.json"
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)
        logging.info(f"Saved run info: {run_info_path}")

    except Exception as e:
        logging.error(f"Dataset evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()