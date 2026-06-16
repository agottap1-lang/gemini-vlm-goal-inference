#!/usr/bin/env python3
"""
Compare Human vs VLM Performance in Terms of FRAMES
====================================================

This analysis converts time-based metrics to frame-based metrics
to understand how many frames each agent needs to make correct decisions.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
PARTICIPANT_DATA_DIR = Path(r"C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\participant_data")
VLM_OUTPUTS_DIR = Path("outputs")
USER_STUDY_TIMEPOINTS_FILE = Path("user_study_timepoints.csv")
OUTPUT_DIR = Path("analysis_results_final")

VIDEO_ID_MAPPING = {"amb_d_drawer_cclose": "amb_d_drawer_close"}

# Video frame rates (frames per second) - typical is 30 fps
FPS = 30

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

TRAJECTORY_TYPES = {
    'amb_d_drawer_close': 'ambiguous',
    'amb_l_block': 'ambiguous',
    'amb_r_block': 'ambiguous',
    'amb_to_drawer_close': 'ambiguous',
    'le_d_drawer_close': 'legible',
    'le_l_block': 'legible',
    'le_r_block': 'legible',
    'le_t_drawer_close': 'legible',
}

def seconds_to_frames(seconds, fps=FPS):
    """Convert seconds to frame number."""
    return int(seconds * fps)

def load_user_study_timepoints():
    """Load study timepoints."""
    df = pd.read_csv(USER_STUDY_TIMEPOINTS_FILE)
    timepoints_map = {}
    for _, row in df.iterrows():
        video_id = row['video_id']
        timepoints_str = row['recommended_timepoints']
        timepoints = [int(t.strip()) for t in timepoints_str.split(',')]
        timepoints_map[video_id] = timepoints
    return timepoints_map

def find_latest_vlm_run(video_id):
    """Find latest VLM run."""
    video_dir = VLM_OUTPUTS_DIR / video_id
    run_dirs = sorted(video_dir.glob("run_*_prefix"))
    if not run_dirs:
        raise FileNotFoundError(f"No runs for {video_id}")
    return run_dirs[-1] / "results.jsonl"

def analyze_frame_performance():
    """Analyze performance in terms of frames."""
    
    print("=" * 90)
    print("FRAME-BASED PERFORMANCE COMPARISON: HUMAN vs VLM")
    print("=" * 90)
    print(f"\nAssuming video frame rate: {FPS} fps")
    print()
    
    results = []
    
    for video_id in GROUND_TRUTH.keys():
        gt = GROUND_TRUTH[video_id]
        traj_type = TRAJECTORY_TYPES[video_id]
        
        print(f"=== {video_id} (Ground Truth: {gt}) ===")
        
        # Load VLM predictions
        try:
            vlm_file = find_latest_vlm_run(video_id)
            vlm_preds = []
            with open(vlm_file) as f:
                for line in f:
                    if line.strip():
                        vlm_preds.append(json.loads(line))
            vlm_preds = sorted(vlm_preds, key=lambda x: x['t_sec'])
            
            # Find first correct VLM prediction
            vlm_first_correct = None
            vlm_first_correct_frame = None
            for pred in vlm_preds:
                if pred['choice'] == gt:
                    vlm_first_correct = pred['t_sec']
                    vlm_first_correct_frame = seconds_to_frames(vlm_first_correct)
                    break
            
            if vlm_first_correct is not None:
                print(f"  VLM:   First correct at t={vlm_first_correct}s → Frame {vlm_first_correct_frame}")
            else:
                print(f"  VLM:   Never correct")
        except Exception as e:
            print(f"  VLM:   Error - {e}")
            vlm_first_correct = None
            vlm_first_correct_frame = None
        
        # Load human decisions
        human_times = []
        human_frames = []
        human_correct_count = 0
        human_total = 0
        
        for json_file in sorted(PARTICIPANT_DATA_DIR.glob("*.json")):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for obs in data:
                        if isinstance(obs, dict):
                            vid = obs.get('video_id')
                            vid = VIDEO_ID_MAPPING.get(vid, vid)
                            
                            if vid == video_id and obs.get('phase') == 'video_stop':
                                decision_time = obs.get('decision_time_sec')
                                choice = obs.get('choice', '').upper()
                                
                                if decision_time is not None:
                                    human_total += 1
                                    if choice == gt:
                                        human_times.append(decision_time)
                                        human_frames.append(seconds_to_frames(decision_time))
                                        human_correct_count += 1
            except Exception:
                continue
        
        if human_times:
            human_mean_time = sum(human_times) / len(human_times)
            human_mean_frame = sum(human_frames) / len(human_frames)
            print(f"  Human: Mean decision at t={human_mean_time:.2f}s → Frame {human_mean_frame:.0f}")
            print(f"         ({human_correct_count}/{human_total} participants correct)")
        else:
            human_mean_time = None
            human_mean_frame = None
            print(f"  Human: No correct decisions")
        
        # Calculate frame difference
        if vlm_first_correct_frame is not None and human_mean_frame is not None:
            frame_diff = vlm_first_correct_frame - human_mean_frame
            frames_saved = -frame_diff
            print(f"  Difference: VLM {frame_diff:+.0f} frames ({frame_diff/FPS:+.2f}s)")
            if frame_diff < 0:
                print(f"  → VLM needs {abs(frames_saved):.0f} FEWER frames ({abs(frame_diff/FPS):.2f}s faster)")
            else:
                print(f"  → Human needs {frames_saved:.0f} FEWER frames ({frame_diff/FPS:.2f}s faster)")
        
        results.append({
            'video_id': video_id,
            'trajectory_type': traj_type,
            'ground_truth': gt,
            'vlm_time_sec': vlm_first_correct,
            'vlm_frame': vlm_first_correct_frame,
            'human_mean_time_sec': human_mean_time,
            'human_mean_frame': human_mean_frame,
            'frame_difference': vlm_first_correct_frame - human_mean_frame if (vlm_first_correct_frame and human_mean_frame) else None,
            'vlm_fewer_frames': (human_mean_frame - vlm_first_correct_frame) if (vlm_first_correct_frame and human_mean_frame) else None
        })
        
        print()
    
    # Summary
    print("=" * 90)
    print("SUMMARY: FRAME EFFICIENCY")
    print("=" * 90)
    print()
    
    df = pd.DataFrame(results)
    
    # Overall stats
    valid = df[df['frame_difference'].notna()]
    
    print(f"Videos analyzed: {len(valid)}")
    print(f"VLM faster (fewer frames needed): {(valid['frame_difference'] < 0).sum()}/{len(valid)} videos")
    print()
    
    # By trajectory type
    for traj_type in ['legible', 'ambiguous']:
        traj_df = valid[valid['trajectory_type'] == traj_type]
        if len(traj_df) > 0:
            mean_frame_diff = traj_df['frame_difference'].mean()
            vlm_faster_count = (traj_df['frame_difference'] < 0).sum()
            print(f"{traj_type.capitalize()} Trajectories:")
            print(f"  Mean frame difference: {mean_frame_diff:+.1f} frames ({mean_frame_diff/FPS:+.2f}s)")
            print(f"  VLM faster: {vlm_faster_count}/{len(traj_df)} videos")
            print(f"  Frame savings: {-mean_frame_diff:.0f} frames on average")
            print()
    
    # Detailed table
    print("=" * 90)
    print("DETAILED FRAME COMPARISON TABLE")
    print("=" * 90)
    print()
    print(f"{'Video':<25} {'VLM Frame':<12} {'Human Frame':<14} {'VLM Saves':<12}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        vlm_f = f"{row['vlm_frame']:.0f}" if pd.notna(row['vlm_frame']) else "N/A"
        human_f = f"{row['human_mean_frame']:.0f}" if pd.notna(row['human_mean_frame']) else "N/A"
        saves = f"{row['vlm_fewer_frames']:+.0f}" if pd.notna(row['vlm_fewer_frames']) else "N/A"
        
        print(f"{row['video_id']:<25} {vlm_f:<12} {human_f:<14} {saves:<12}")
    
    print()
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "frame_based_comparison.csv", index=False)
    print(f"✓ Saved results to {OUTPUT_DIR / 'frame_based_comparison.csv'}")
    print()
    
    # Create visualization
    create_frame_comparison_figure(df)
    
    return df

def create_frame_comparison_figure(df):
    """Create publication-ready figure for frame-based comparison."""
    
    print("=" * 90)
    print("CREATING FRAME-BASED COMPARISON FIGURE")
    print("=" * 90)
    print()
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'serif'
    
    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter valid data
    valid_df = df[df['frame_difference'].notna()].copy()
    
    # Panel A: Frame count comparison
    ax = axes[0]
    x = np.arange(len(valid_df))
    width = 0.35
    
    vlm_frames = valid_df['vlm_frame'].values
    human_frames = valid_df['human_mean_frame'].values
    
    ax.bar(x - width/2, human_frames, width, label='Human', 
           color='#2E7D32', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x + width/2, vlm_frames, width, label='VLM', 
           color='#FFA726', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Video', fontweight='bold', fontsize=12)
    ax.set_ylabel('Frames to Correct Inference', fontweight='bold', fontsize=12)
    ax.set_title('(A) Frame-Based Performance Comparison', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in valid_df['video_id']], 
                        rotation=0, fontsize=9)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, (h_val, v_val) in enumerate(zip(human_frames, vlm_frames)):
        ax.text(i - width/2, h_val + 5, f'{int(h_val)}', 
                ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, v_val + 5, f'{int(v_val)}', 
                ha='center', va='bottom', fontsize=8)
    
    # Panel B: Frame difference (VLM advantage)
    ax = axes[1]
    frame_diffs = valid_df['vlm_fewer_frames'].values  # Positive = VLM advantage
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in frame_diffs]
    
    bars = ax.bar(range(len(valid_df)), frame_diffs, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Video', fontweight='bold', fontsize=12)
    ax.set_ylabel('Frame Advantage (frames)', fontweight='bold', fontsize=12)
    ax.set_title('(B) VLM Frame Savings\n(Positive = VLM Better)', 
                 fontweight='bold', fontsize=13)
    ax.set_xticks(range(len(valid_df)))
    ax.set_xticklabels([v.replace('_', '\n') for v in valid_df['video_id']], 
                        rotation=0, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, val in enumerate(frame_diffs):
        y_pos = val + (5 if val > 0 else -5)
        va = 'bottom' if val > 0 else 'top'
        ax.text(i, y_pos, f'{int(val):+d}', 
                ha='center', va=va, fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(OUTPUT_DIR / "figure3_frame_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure3_frame_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved figure to {OUTPUT_DIR / 'figure3_frame_comparison.png/pdf'}")
    print()

if __name__ == "__main__":
    analyze_frame_performance()
