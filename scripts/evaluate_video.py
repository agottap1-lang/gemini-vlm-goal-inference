#!/usr/bin/env python3
"""
Single-video evaluation pipeline with full provenance capture.

RECOMMENDED FOR: Testing a single video with detailed output including run_info.json.
FOR BATCH PROCESSING: Use scripts/eval_dataset.py instead.

Produces: outputs/{video_id}_all.jsonl, reports/{video_id}_analysis.md, 
          outputs/run_info.json, outputs/pip_freeze.txt
"""

import sys
import os
import json
import logging
import uuid
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from importlib import metadata as importlib_metadata

import cv2

# ============================================================================
# CONFIGURATION: Edit these lines for your video
# ============================================================================
VIDEO_ID = "le_r_block"  # e.g., "amb_l_block", "le_l_block", etc.
VIDEO_PATH = r"videos/le r block.mp4"  # Relative or absolute path to video

# EVALUATION MODE
MODE = "prefix_frames"  # Options: "single_frame" or "prefix_frames"
SAVE_FRAMES = True      # Set to True to save frames to outputs/frames/{video_id}/t_{t_sec:03d}.png

# ============================================================================

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gemini_vlm_eval.video import extract_frames, extract_and_cache_frames
from src.gemini_vlm_eval.client import GeminiClient
from src.gemini_vlm_eval.schema import EvaluationResult, ManifestEntry
from src.gemini_vlm_eval.prompt import get_instruction_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_manifest():
    """Load manifest.jsonl to get video metadata."""
    manifest_path = Path(__file__).parent.parent / "data" / "manifest.jsonl"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                manifest[entry['video_id']] = entry
    return manifest


def get_git_info(repo_root: Path) -> tuple[str, bool]:
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


def evaluate_video(video_id: str, video_path: str, manifest: dict, mode: str = "single_frame", save_frames: bool = False, output_dir: Path = None) -> list:
    """
    Evaluate all seconds of a video and return EvaluationResult list.
    
    Args:
        video_id: Video identifier
        video_path: Path to video file
        manifest: Manifest dictionary
        mode: Evaluation mode - "single_frame" or "prefix_frames"
        save_frames: If True, save frames to output_dir/frames/{video_id}/t_{t_sec:03d}.png
        output_dir: Directory to save frames (if None, uses outputs/ directory)
    
    Returns:
        List of EvaluationResult objects
    """
    
    # Load metadata from manifest if available
    meta = manifest.get(video_id, {})
    goal_a = meta.get('goal_A', 'Goal A')
    goal_b = meta.get('goal_B', 'Goal B')
    goal_gt = meta.get('goal_gt', 'A')
    scene_id = meta.get('scene_id', meta.get('scene', 'unknown'))
    task_family = meta.get('task_family', 'unknown')
    traj_type = meta.get('traj_type', 'unknown')
    notes = meta.get('notes', '')
    resolved_video_path = meta.get('video_path', video_path)
    
    # Determine frames directory
    if output_dir and save_frames:
        frames_base_dir = output_dir / "frames"
    else:
        frames_base_dir = Path(__file__).parent.parent / "outputs" / "frames"
    
    # Extract and cache frames
    frames_dict = extract_and_cache_frames(
        video_path=video_path,
        video_id=video_id,
        sample_rate_seconds=1.0,
        max_frames=None,  # All frames
        save_frames=save_frames,
        output_dir=frames_base_dir if save_frames else None
    )
    
    timestamps = sorted(frames_dict.keys())
    logger.info(f"Extracted {len(timestamps)} frames at 1-second intervals from {video_id}")
    
    client = GeminiClient()
    results = []
    for t_sec in timestamps:
        # Prepare image data based on mode
        if mode == "prefix_frames":
            # Send all frames from 0 to t_sec (inclusive)
            prefix_frames = [frames_dict[t]["jpeg_bytes"] for t in range(0, t_sec + 1) if t in frames_dict]
            image_data = prefix_frames
        else:  # single_frame
            image_data = frames_dict[t_sec]["jpeg_bytes"]
        
        manifest_entry = ManifestEntry(
            video_id=video_id,
            video_path=resolved_video_path,
            goal_gt=goal_gt,
            goal_A=goal_a,
            goal_B=goal_b,
            scene_id=scene_id,
            task_family=task_family,
            traj_type=traj_type,
            notes=notes,
        )

        result = client.evaluate_frame(
            image_bytes=image_data,
            manifest_entry=manifest_entry,
            t_sec=t_sec,
            frame_idx=frames_dict[t_sec]["frame_idx"],
            mode=mode
        )
        results.append(result)
        logger.info(f"{video_id} t={t_sec:.0f}s: choice={result.choice}, confidence={result.confidence}")
    
    return results


def save_jsonl(results: list, output_path: Path):
    """Save EvaluationResult list to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(result.model_dump_json() + '\n')
    logger.info(f"Saved JSONL: {output_path}")


def generate_analysis_markdown(results: list, manifest: dict, video_id: str) -> str:
    """Generate markdown analysis from results."""
    
    meta = manifest.get(video_id, {})
    goal_a = meta.get('goal_A', 'Goal A')
    goal_b = meta.get('goal_B', 'Goal B')
    scene = meta.get('scene', 'Unknown')
    traj_type = meta.get('traj_type', 'Unknown')
    video_path = meta.get('video_path', 'Unknown')
    
    # Statistics
    choice_counts = {'A': 0, 'B': 0, 'C': 0}
    legible_count = 0
    confidences = []
    high_conf_moments = []
    
    for r in results:
        choice_counts[r.choice] += 1
        if r.legible == "legible_now":
            legible_count += 1
        confidences.append(r.confidence)
        if r.confidence >= 95:
            high_conf_moments.append((r.t_sec, r.choice, r.confidence, r.pA, r.pB))
    
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    max_conf = max(confidences) if confidences else 0
    min_conf = min(confidences) if confidences else 0
    
    # Build markdown
    md = []
    md.append(f"# {video_id}")
    md.append(f"**Path:** {video_path}")
    md.append(f"**Goal A:** {goal_a}")
    md.append(f"**Goal B:** {goal_b}")
    md.append(f"**Scene:** {scene}")
    md.append(f"**Trajectory:** {traj_type}")
    md.append("")
    
    # Statistics section
    md.append("## Statistics")
    md.append(f"- **Duration:** {len(results)} seconds")
    md.append(f"- **Predicted A:** {choice_counts['A']}")
    md.append(f"- **Predicted B:** {choice_counts['B']}")
    md.append(f"- **Uncertain (C):** {choice_counts['C']}")
    md.append(f"- **Legible moments:** {legible_count}")
    md.append(f"- **Avg confidence:** {avg_conf:.1f}%")
    md.append(f"- **Max confidence:** {max_conf}%")
    md.append(f"- **Min confidence:** {min_conf}%")
    md.append("")
    
    # High-confidence moments
    if high_conf_moments:
        md.append("## High-Confidence Moments (≥95%)")
        for t, choice, conf, pA, pB in high_conf_moments:
            md.append(f"- **t={t}s:** choice={choice}, confidence={conf}% (pA={pA:.2f}, pB={pB:.2f})")
        md.append("")
    
    # Frame-by-frame table
    md.append("## Frame-by-Frame Analysis")
    md.append("| Time (s) | Choice | Confidence | pA | pB | Cue | Legible |")
    md.append("|----------|--------|------------|----|----|-----|---------|")
    for r in results:
        cue_text = r.cue[:30] if r.cue else "—"
        md.append(f"| {r.t_sec} | {r.choice} | {r.confidence}% | {r.pA:.2f} | {r.pB:.2f} | {cue_text} | {str(r.legible)[:1]} |")
    md.append("")
    
    # Interpretation
    md.append("## Interpretation")
    if traj_type.lower() == "legible":
        avg_a = choice_counts['A'] / len(results) if results else 0
        if avg_a > 0.5:
            md.append("✓ **Strong legible trajectory.** Model consistently identifies goal A with high confidence during key moments.")
        else:
            md.append("⚠ **Mixed signals.** Despite being labeled legible, model shows uncertainty in goal identification.")
    elif traj_type.lower() == "ambiguous":
        uncertain = choice_counts['C'] / len(results) if results else 0
        if uncertain > 0.5:
            md.append("✓ **As expected ambiguous.** Model frequently outputs uncertain choices (C), reflecting inherent ambiguity.")
        else:
            md.append("⚠ **Model found legible cues.** Despite ambiguity label, model identified distinguishing features.")
    
    return '\n'.join(md)


def main():
    run_id = str(uuid.uuid4())
    timestamp_utc_start = datetime.utcnow().isoformat() + "Z"
    cli_command = " ".join(sys.argv) if sys.argv else ""
    repo_root = Path(__file__).parent.parent

    # Validation
    video_path_obj = Path(VIDEO_PATH)
    if not video_path_obj.exists():
        logger.error(f"Video not found: {VIDEO_PATH}")
        sys.exit(1)
    
    # Create timestamped run directory for this video
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    mode_short = MODE.replace("_", "")[:6]  # "single" or "prefix"
    run_dir_name = f"run_{timestamp_str}_{mode_short}"
    
    # Organize outputs by video_id
    video_output_dir = Path(__file__).parent.parent / "outputs" / VIDEO_ID / run_dir_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting evaluation: {VIDEO_ID}")
    logger.info(f"Video path: {VIDEO_PATH}")
    logger.info(f"Mode: {MODE}")
    logger.info(f"Save frames: {SAVE_FRAMES}")
    logger.info(f"Output directory: {video_output_dir}")
    
    # Load manifest
    manifest = load_manifest()
    
    # Evaluate video (frames will be saved to video_output_dir/frames if SAVE_FRAMES=True)
    results = evaluate_video(VIDEO_ID, VIDEO_PATH, manifest, mode=MODE, save_frames=SAVE_FRAMES, output_dir=video_output_dir)
    
    # Save JSONL to run directory
    jsonl_path = video_output_dir / "results.jsonl"
    save_jsonl(results, jsonl_path)
    
    # Generate and save markdown to reports/{video_id}/ directory
    markdown = generate_analysis_markdown(results, manifest, VIDEO_ID)
    report_dir = Path(__file__).parent.parent / "reports" / VIDEO_ID
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"analysis_{timestamp_str}_{mode_short}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    logger.info(f"Saved report: {report_path}")

    # Persist run info to run directory
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

    run_info = {
        "run_id": run_id,
        "run_directory": str(video_output_dir),
        "timestamp_utc_start": timestamp_utc_start,
        "timestamp_utc_end": timestamp_utc_end,
        "video_id": VIDEO_ID,
        "video_path": VIDEO_PATH,
        "git_commit": git_commit,
        "repo_dirty": repo_dirty,
        "python_version": python_version,
        "os": os_name,
        "platform": platform_str,
        "opencv_version": opencv_version,
        "google_genai_version": google_genai_version,
        "evaluation_mode": MODE,
        "save_frames": SAVE_FRAMES,
        "command_line": cli_command,
        "api_key_source": "env:GEMINI_API_KEY",
    }

    # pip freeze capture to run directory
    pip_freeze_path = video_output_dir / "pip_freeze.txt"
    write_pip_freeze(pip_freeze_path)
    run_info["pip_freeze_path"] = str(pip_freeze_path)

    # Optional machine info
    run_info["machine_info"] = get_optional_machine_info()

    # Save run_info to run directory
    run_info_path = video_output_dir / "run_info.json"
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    logger.info(f"Saved run info: {run_info_path}")
    
    logger.info("✓ Done!")
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE: {VIDEO_ID}")
    print(f"{'='*70}")
    print(f"Mode: {MODE}")
    print(f"Run directory: {video_output_dir}")
    print(f"\nResults saved to:")
    print(f"  📊 JSONL:   {jsonl_path}")
    print(f"  📝 Report:  {report_path}")
    print(f"  📋 Metadata: {run_info_path}")
    if SAVE_FRAMES:
        frames_dir = video_output_dir / "frames"
        print(f"  🖼️  Frames:  {frames_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
