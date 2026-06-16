#!/usr/bin/env python3
"""
DEPRECATED: Use eval_dataset.py for canonical manifest-driven evaluation.

This script was used for batch processing with embedded analysis generation.
For the canonical pipeline, use:
  python scripts/eval_dataset.py --manifest data/manifest.jsonl --k all --out outputs/results.jsonl
  python scripts/analyze_jsonl.py outputs/results.jsonl --output reports/analysis.md
"""

import logging
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.schema import ManifestEntry


def load_manifest(manifest_path: str):
    """Load manifest JSONL."""
    entries = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(ManifestEntry(**json.loads(line)))
    return entries


def evaluate_video_for_k_seconds(client, manifest_entry, k, output_file):
    """Evaluate video frames and write to output file."""
    from gemini_vlm_eval.video import extract_frames
    
    video_path = manifest_entry.video_path
    if not Path(video_path).exists():
        logging.error(f"Video not found: {video_path}")
        return
    
    frames_data = list(extract_frames(video_path, sample_rate_seconds=1.0, max_frames=k))
    
    for frame_idx, frame_bytes, t_sec in frames_data:
        try:
            result = client.evaluate_frame(frame_bytes, manifest_entry, int(t_sec), frame_idx)
            output_file.write(result.model_dump_json() + '\n')
            output_file.flush()
        except Exception as e:
            logging.error(f"Failed to evaluate {manifest_entry.video_id} at t={t_sec}s: {e}")


def load_results(jsonl_path: str):
    """Load JSONL results."""
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def generate_markdown_table(results):
    """Generate markdown table."""
    lines = []
    lines.append("| t_sec | pA/pB | choice | confidence | legible | cue |")
    lines.append("|-------|-------|--------|------------|---------|-----|")
    
    for r in results:
        t_sec = r.get('t_sec', '?')
        pA = r.get('pA', 0.0)
        pB = r.get('pB', 0.0)
        choice = r.get('choice', '?')
        confidence = r.get('confidence', '?')
        legible = r.get('legible', '?')
        cue = r.get('cue', '')[:50]
        
        lines.append(f"| {t_sec} | {pA:.2f}/{pB:.2f} | {choice} | {confidence} | {legible} | {cue} |")
    
    return "\n".join(lines)


def generate_analysis(results):
    """Generate analysis summary."""
    if not results:
        return "No results to analyze."
    
    lines = []
    first = results[0]
    video_id = first.get('video_id', 'unknown')
    video_path = first.get('video_path', 'unknown')
    goal_gt = first.get('goal_gt', '?')
    goal_A = first.get('goal_A', '?')
    goal_B = first.get('goal_B', '?')
    traj_type = first.get('traj_type', 'unknown')
    
    lines.append(f"**Video:** {video_id}")
    lines.append(f"**Path:** {video_path}")
    lines.append(f"**Ground Truth:** Goal {goal_gt} ({goal_A if goal_gt == 'A' else goal_B})")
    lines.append(f"**Trajectory Type:** {traj_type}")
    lines.append(f"**Duration:** ~{len(results)} seconds (1fps sampling)")
    lines.append("")
    
    choices = [r.get('choice') for r in results]
    confidences = [r.get('confidence', 0) for r in results]
    legibles = [r.get('legible') for r in results]
    
    choice_A_count = choices.count('A')
    choice_B_count = choices.count('B')
    choice_C_count = choices.count('C')
    legible_now_count = legibles.count('legible_now')
    
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    max_conf = max(confidences) if confidences else 0
    min_conf = min(confidences) if confidences else 0
    
    lines.append("**Statistics:**")
    lines.append(f"- Predicted A: {choice_A_count} frames")
    lines.append(f"- Predicted B: {choice_B_count} frames")
    lines.append(f"- Uncertain (C): {choice_C_count} frames")
    lines.append(f"- Legible moments: {legible_now_count} frames")
    lines.append(f"- Avg confidence: {avg_conf:.1f}%")
    lines.append(f"- Max confidence: {max_conf}%")
    lines.append(f"- Min confidence: {min_conf}%")
    lines.append("")
    
    high_conf = [r for r in results if r.get('confidence', 0) >= 70]
    if high_conf:
        lines.append("**High-Confidence Moments (>= 70%):**")
        for r in high_conf:
            t = r.get('t_sec')
            choice = r.get('choice')
            conf = r.get('confidence')
            cue = r.get('cue', '')[:50]
            lines.append(f"- t={t}s: choice={choice}, confidence={conf}% -> {cue}")
        lines.append("")
    
    lines.append("**Interpretation:**")
    if traj_type == "legible":
        if choice_A_count >= choice_B_count:
            lines.append(f"Strong legible trajectory. Model consistently identifies goal A with high confidence during key moments.")
        else:
            lines.append(f"Legible trajectory toward goal B detected with high confidence peaks.")
    elif traj_type == "ambiguous":
        if choice_C_count > (choice_A_count + choice_B_count) / 2:
            lines.append(f"Ambiguous trajectory confirmed: model frequently uncertain (choice=C). Goal identification is challenging throughout.")
        else:
            lines.append(f"Ambiguous trajectory with sporadic goal signals. Model shows weak preference with intermittent confidence peaks.")
    else:
        lines.append("Generic trajectory analysis.")
    
    return "\n".join(lines)



def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    manifest_path = "data/manifest.jsonl"
    entries = load_manifest(manifest_path)
    
    # Create output directories
    Path("outputs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    client = GeminiClient()
    
    for i, entry in enumerate(entries, 1):
        video_id = entry.video_id
        print(f"\n[{i}/{len(entries)}] Evaluating {video_id}...")
        
        # Evaluate video
        out_jsonl = Path(f"outputs/{video_id}_all.jsonl")
        try:
            with open(out_jsonl, 'w') as f:
                evaluate_video_for_k_seconds(client, entry, None, f)
            print(f"  ✓ Saved: {out_jsonl}")
        except Exception as e:
            logging.error(f"Failed to evaluate {video_id}: {e}")
            continue
        
        # Generate analysis report
        try:
            results = load_results(str(out_jsonl))
            analysis = generate_analysis(results)
            table = generate_markdown_table(results)
            report = f"{analysis}\n\n{table}"
            
            out_md = Path(f"reports/{video_id}_analysis.md")
            with open(out_md, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"  ✓ Saved: {out_md}")
        except Exception as e:
            logging.error(f"Failed to analyze {video_id}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Complete! Evaluated {len(entries)} videos.")
    print(f"JSONL outputs: outputs/*_all.jsonl")
    print(f"Reports: reports/*_analysis.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
