#!/usr/bin/env python3
"""
Run VLM evaluation with Gemini Pro model and compare with Flash results.
Identifies worst-performing videos and shows improvements.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# API key is loaded from .env file via GeminiClient

from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.schema import ManifestEntry
from gemini_vlm_eval.video import extract_frames

# Configuration
MANIFEST_FILE = Path("data/manifest.jsonl")
USER_STUDY_TIMEPOINTS_FILE = Path("user_study_timepoints.csv")
OUTPUTS_DIR = Path("outputs")
ANALYSIS_DIR = Path("analysis_results_pro_comparison")

# Models to compare
MODEL_PRO = "gemini-2.0-flash-exp"  # Pro model (experimental)
MODEL_FLASH = "gemini-2.5-flash"  # Current model

# Ground truth
GROUND_TRUTH = {
    'amb_d_drawer_close': 'A',
    'amb_l_block': 'A',
    'amb_r_block': 'B',
    'amb_to_drawer_close': 'B',
    'le_d_drawer_close': 'A',
    'le_l_block': 'A',
    'le_r_block': 'B',
    'le_t_drawer_close': 'B',
}

def load_user_study_timepoints():
    """Load timepoints where humans were evaluated."""
    import pandas as pd
    df = pd.read_csv(USER_STUDY_TIMEPOINTS_FILE)
    timepoints_map = {}
    for _, row in df.iterrows():
        video_id = row['video_id']
        timepoints_str = row['recommended_timepoints']
        timepoints = [int(t.strip()) for t in timepoints_str.split(',')]
        timepoints_map[video_id] = timepoints
    return timepoints_map

def load_manifest():
    """Load manifest with video metadata."""
    manifest = []
    with open(MANIFEST_FILE, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                manifest.append(ManifestEntry(**data))
    return manifest

def evaluate_video_at_timepoints(video_entry, timepoints, model_name):
    """Evaluate a video at specific timepoints using given model."""
    print(f"  Evaluating {video_entry.video_id} at timepoints {timepoints}")
    
    # Extract frames
    frame_data = list(extract_frames(video_entry.video_path, sample_rate_seconds=1.0))
    frame_bytes_dict = {int(t): jpeg_bytes for _, jpeg_bytes, t in frame_data}
    
    # Initialize client
    client = GeminiClient(model=model_name)
    
    results = []
    for t in timepoints:
        if t not in frame_bytes_dict:
            print(f"    Warning: No frame at t={t}s")
            continue
        
        # Prepare frames (prefix mode: all from 0 to t)
        frames_to_send = [frame_bytes_dict[i] for i in range(t + 1) if i in frame_bytes_dict]
        
        try:
            result = client.evaluate_frame(
                image_bytes=frames_to_send,
                manifest_entry=video_entry,
                t_sec=t,
                frame_idx=t,
                mode="prefix_frames"
            )
            results.append(result)
            print(f"    t={t}s: choice={result.choice}, confidence={result.confidence}%")
        except Exception as e:
            print(f"    t={t}s: ERROR - {e}")
    
    return results

def calculate_accuracy(results):
    """Calculate accuracy for VLM predictions."""
    correct = 0
    total = 0
    
    for result in results:
        if result.choice in ['A', 'B']:  # Only count non-uncertain predictions
            total += 1
            if result.choice == result.goal_gt:
                correct += 1
    
    return (correct / total * 100) if total > 0 else 0.0

def load_existing_flash_results():
    """Load existing Flash model results from output directories."""
    flash_results = []
    
    for video_id in GROUND_TRUTH.keys():
        video_dir = OUTPUTS_DIR / video_id
        if not video_dir.exists():
            continue
        
        # Find latest prefix run
        prefix_runs = sorted(video_dir.glob("run_*_prefix"))
        if not prefix_runs:
            continue
        
        latest_run = prefix_runs[-1]
        results_file = latest_run / "results.jsonl"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        flash_results.append(data)
    
    return flash_results

def compare_models():
    """Run Pro model evaluation and compare with Flash results."""
    
    print("=" * 90)
    print("GEMINI PRO vs FLASH MODEL COMPARISON")
    print("=" * 90)
    print()
    
    # Load data
    print("Loading configuration...")
    manifest = load_manifest()
    timepoints_map = load_user_study_timepoints()
    print(f"✓ Loaded {len(manifest)} videos")
    print()
    
    # Load existing Flash results
    print("Loading existing Flash results...")
    flash_results = load_existing_flash_results()
    print(f"✓ Loaded {len(flash_results)} Flash predictions")
    print()
    
    # Filter Flash results to study timepoints
    flash_filtered = []
    for result in flash_results:
        video_id = result['video_id']
        t_sec = result['t_sec']
        if t_sec in timepoints_map.get(video_id, []):
            flash_filtered.append(result)
    
    print(f"✓ Filtered to {len(flash_filtered)} Flash predictions at study timepoints")
    print()
    
    # Run Pro model evaluation
    print("=" * 90)
    print(f"RUNNING PRO MODEL EVALUATION ({MODEL_PRO})")
    print("=" * 90)
    print()
    
    ANALYSIS_DIR.mkdir(exist_ok=True)
    pro_results = []
    
    for video_entry in manifest:
        video_id = video_entry.video_id
        timepoints = timepoints_map.get(video_id, [])
        
        if not timepoints:
            print(f"⚠️  No timepoints for {video_id}, skipping")
            continue
        
        print(f"📹 {video_id}")
        results = evaluate_video_at_timepoints(video_entry, timepoints, MODEL_PRO)
        pro_results.extend(results)
        print()
    
    # Save Pro results
    pro_results_file = ANALYSIS_DIR / "pro_model_results.jsonl"
    with open(pro_results_file, 'w') as f:
        for result in pro_results:
            f.write(json.dumps(result.model_dump()) + '\n')
    print(f"✓ Saved Pro results to {pro_results_file}")
    print()
    
    # Calculate metrics
    print("=" * 90)
    print("COMPARISON RESULTS")
    print("=" * 90)
    print()
    
    # Overall accuracy
    flash_correct = sum(1 for r in flash_filtered if r['choice'] == GROUND_TRUTH.get(r['video_id']))
    flash_total = len([r for r in flash_filtered if r['choice'] in ['A', 'B']])
    flash_accuracy = (flash_correct / flash_total * 100) if flash_total > 0 else 0
    
    pro_correct = sum(1 for r in pro_results if r.choice == r.goal_gt and r.choice in ['A', 'B'])
    pro_total = len([r for r in pro_results if r.choice in ['A', 'B']])
    pro_accuracy = (pro_correct / pro_total * 100) if pro_total > 0 else 0
    
    print(f"Overall Accuracy:")
    print(f"  Flash ({MODEL_FLASH}): {flash_accuracy:.2f}% ({flash_correct}/{flash_total})")
    print(f"  Pro ({MODEL_PRO}):     {pro_accuracy:.2f}% ({pro_correct}/{pro_total})")
    print(f"  Improvement:           {pro_accuracy - flash_accuracy:+.2f}%")
    print()
    
    # Per-video accuracy
    print("Per-Video Accuracy:")
    print(f"{'Video':<25} {'Flash':<12} {'Pro':<12} {'Change':<10}")
    print("-" * 90)
    
    video_comparison = []
    
    for video_id in sorted(GROUND_TRUTH.keys()):
        gt = GROUND_TRUTH[video_id]
        
        # Flash accuracy
        flash_vid = [r for r in flash_filtered if r['video_id'] == video_id and r['choice'] in ['A', 'B']]
        flash_vid_correct = sum(1 for r in flash_vid if r['choice'] == gt)
        flash_vid_acc = (flash_vid_correct / len(flash_vid) * 100) if flash_vid else 0
        
        # Pro accuracy
        pro_vid = [r for r in pro_results if r.video_id == video_id and r.choice in ['A', 'B']]
        pro_vid_correct = sum(1 for r in pro_vid if r.choice == gt)
        pro_vid_acc = (pro_vid_correct / len(pro_vid) * 100) if pro_vid else 0
        
        change = pro_vid_acc - flash_vid_acc
        
        print(f"{video_id:<25} {flash_vid_acc:>6.1f}%      {pro_vid_acc:>6.1f}%      {change:>+6.1f}%")
        
        video_comparison.append({
            'video_id': video_id,
            'flash_accuracy': flash_vid_acc,
            'pro_accuracy': pro_vid_acc,
            'change': change,
            'flash_correct': flash_vid_correct,
            'flash_total': len(flash_vid),
            'pro_correct': pro_vid_correct,
            'pro_total': len(pro_vid)
        })
    
    print()
    
    # Identify worst-performing videos
    print("=" * 90)
    print("WORST-PERFORMING VIDEOS (Pro Model)")
    print("=" * 90)
    print()
    
    sorted_videos = sorted(video_comparison, key=lambda x: x['pro_accuracy'])
    
    print(f"{'Rank':<6} {'Video':<25} {'Accuracy':<12} {'Correct/Total'}")
    print("-" * 90)
    for i, vid in enumerate(sorted_videos[:5], 1):
        print(f"{i:<6} {vid['video_id']:<25} {vid['pro_accuracy']:>6.1f}%      {vid['pro_correct']}/{vid['pro_total']}")
    
    print()
    
    # Videos with biggest improvement
    print("=" * 90)
    print("BIGGEST IMPROVEMENTS (Flash → Pro)")
    print("=" * 90)
    print()
    
    sorted_improvements = sorted(video_comparison, key=lambda x: x['change'], reverse=True)
    
    print(f"{'Rank':<6} {'Video':<25} {'Improvement':<12} {'Flash→Pro'}")
    print("-" * 90)
    for i, vid in enumerate(sorted_improvements[:5], 1):
        print(f"{i:<6} {vid['video_id']:<25} {vid['change']:>+6.1f}%      {vid['flash_accuracy']:.1f}%→{vid['pro_accuracy']:.1f}%")
    
    print()
    
    # Save comparison report
    import pandas as pd
    comparison_df = pd.DataFrame(video_comparison)
    comparison_df.to_csv(ANALYSIS_DIR / "model_comparison.csv", index=False)
    print(f"✓ Saved comparison to {ANALYSIS_DIR / 'model_comparison.csv'}")
    
    # Summary stats
    summary = {
        'timestamp': datetime.now().isoformat(),
        'flash_model': MODEL_FLASH,
        'pro_model': MODEL_PRO,
        'flash_accuracy': flash_accuracy,
        'pro_accuracy': pro_accuracy,
        'improvement': pro_accuracy - flash_accuracy,
        'flash_predictions': len(flash_filtered),
        'pro_predictions': len(pro_results),
        'worst_videos': [v['video_id'] for v in sorted_videos[:3]],
        'best_improvements': [v['video_id'] for v in sorted_improvements[:3]]
    }
    
    with open(ANALYSIS_DIR / "comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary to {ANALYSIS_DIR / 'comparison_summary.json'}")
    print()
    
    print("=" * 90)
    print("COMPARISON COMPLETE")
    print("=" * 90)
    
    return summary

if __name__ == "__main__":
    try:
        summary = compare_models()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
