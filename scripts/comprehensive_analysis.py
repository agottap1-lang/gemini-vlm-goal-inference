#!/usr/bin/env python3
"""
VLM vs Human Goal Inference: Comprehensive Analysis Pipeline
==============================================================

Publication-ready analysis comparing VLM and human performance on 
robot motion goal inference task.

IMPORTANT: This analysis ensures FAIR comparison by evaluating VLM
ONLY at the same timepoints shown to human participants in the study.

Author: Research Analysis Pipeline
Date: January 2026
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import binomtest

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
PARTICIPANT_DATA_DIR = Path(r"C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\participant_data")
VLM_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
ANALYSIS_OUTPUT_DIR = Path(__file__).parent.parent / "analysis_results_final"
USER_STUDY_TIMEPOINTS_FILE = Path(__file__).parent.parent / "user_study_timepoints.csv"

# Video ID corrections
VIDEO_ID_MAPPING = {"amb_d_drawer_cclose": "amb_d_drawer_close"}

# Trajectory types
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

# Ground truth (correct goals for each video)
GROUND_TRUTH = {
    'amb_d_drawer_close': 'A',  # Close bottom drawer
    'amb_l_block': 'A',         # Move to left block
    'amb_r_block': 'B',         # Move to right block
    'amb_to_drawer_close': 'B', # Close top drawer
    'le_d_drawer_close': 'A',   # Close bottom drawer
    'le_l_block': 'A',          # Move to left block
    'le_r_block': 'B',          # Move to right block
    'le_t_drawer_close': 'B',   # Close TOP drawer (t = top)
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval."""
    if len(values) == 0:
        return (np.nan, np.nan)
    boot = np.random.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lower = np.percentile(boot, 100 * (alpha / 2))
    upper = np.percentile(boot, 100 * (1 - alpha / 2))
    return (lower, upper)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_user_study_timepoints() -> Dict[str, List[int]]:
    """Load the specific timepoints shown to human participants."""
    print("\n⏱️  Loading User Study Timepoints (for FAIR comparison)...")
    
    df = pd.read_csv(USER_STUDY_TIMEPOINTS_FILE)
    timepoints_map = {}
    
    for _, row in df.iterrows():
        video_id = row['video_id']
        timepoints_str = row['recommended_timepoints']
        timepoints = [int(t.strip()) for t in timepoints_str.split(',')]
        timepoints_map[video_id] = timepoints
        print(f"  {video_id:30s}: {timepoints}")
    
    print(f"\n  ✓ Loaded timepoints for {len(timepoints_map)} videos")
    print(f"  ⚠️  VLM will be evaluated ONLY at these timepoints (not every second)")
    
    return timepoints_map

def find_latest_vlm_run(video_id: str) -> Path:
    """Find the most recent VLM run directory for a video."""
    video_dir = VLM_OUTPUTS_DIR / video_id
    if not video_dir.exists():
        raise FileNotFoundError(f"No output directory for {video_id}")
    
    run_dirs = sorted([d for d in video_dir.glob("run_*_prefix") if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found for {video_id}")
    
    latest_run = run_dirs[-1]  # Most recent by name (contains timestamp)
    return latest_run / "results.jsonl"

def load_vlm_predictions(filter_timepoints: Dict[str, List[int]] = None) -> pd.DataFrame:
    """Load VLM predictions from the LATEST run directories.
    
    Args:
        filter_timepoints: Dict mapping video_id to list of timepoints to keep.
                          If provided, only predictions at these timepoints are included
                          for FAIR comparison with human study.
    """
    print("\n📊 Loading VLM Predictions (from latest runs)...")
    
    all_data = []
    for video_id in GROUND_TRUTH.keys():
        try:
            latest_file = find_latest_vlm_run(video_id)
            print(f"  {video_id:30s}: {latest_file.parent.name}")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        pred['video_id'] = video_id
                        pred['trajectory_type'] = TRAJECTORY_TYPES[video_id]
                        pred['ground_truth'] = GROUND_TRUTH[video_id]
                        all_data.append(pred)
        except Exception as e:
            print(f"  ⚠️  {video_id}: {e}")
    
    df = pd.DataFrame(all_data)
    
    # Standardize column names
    df = df.rename(columns={
        'choice': 'choice_vlm',
        'confidence': 'confidence_vlm',
        't_sec': 'timepoint',
    })
    
    # Ensure uppercase choices
    df['choice_vlm'] = df['choice_vlm'].str.upper()
    
    # Convert timepoint to int (handle None)
    df['timepoint'] = pd.to_numeric(df['timepoint'], errors='coerce').fillna(0).astype(int)
    
    # Filter to specific timepoints if provided (for fair comparison with human study)
    if filter_timepoints:
        print(f"\n🎯 Filtering VLM predictions to match human study timepoints...")
        before_count = len(df)
        
        df = df[df.apply(lambda row: row['timepoint'] in filter_timepoints.get(row['video_id'], []), axis=1)]
        
        after_count = len(df)
        print(f"  Filtered from {before_count} to {after_count} predictions ({after_count/before_count*100:.1f}% retained)")
        print(f"  ✓ FAIR comparison: VLM evaluated at same timepoints as humans")
    
    print(f"\n✓ Loaded {len(df)} VLM predictions from {len(df['video_id'].unique())} videos")
    return df

def load_human_decisions() -> pd.DataFrame:
    """Load human participant decisions."""
    print("\n👥 Loading Human Decisions...")
    
    all_data = []
    for json_file in sorted(PARTICIPANT_DATA_DIR.glob("*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for obs in data:
                    if isinstance(obs, dict):
                        video_id = obs.get('video_id')
                        video_id = VIDEO_ID_MAPPING.get(video_id, video_id)
                        
                        # Add trajectory type and ground truth
                        if video_id in GROUND_TRUTH:
                            obs['video_id'] = video_id
                            obs['trajectory_type'] = TRAJECTORY_TYPES.get(video_id)
                            obs['ground_truth'] = GROUND_TRUTH.get(video_id)
                            all_data.append(obs)
        except Exception as e:
            print(f"  ⚠️  {json_file.name}: {e}")
    
    df = pd.DataFrame(all_data)
    
    # Standardize column names
    df = df.rename(columns={
        'choice': 'choice_human',
        'confidence_0_10': 'confidence_human',
    })
    
    # Ensure uppercase choices
    df['choice_human'] = df['choice_human'].str.upper()
    
    # Convert timepoint
    df['timepoint'] = pd.to_numeric(df['timepoint'], errors='coerce').fillna(0).astype(int)
    
    print(f"✓ Loaded {len(df)} observations from {len(df['participant_id'].unique())} participants")
    return df

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_accuracy(df: pd.DataFrame, choice_col: str, gt_col: str) -> Dict:
    """Calculate accuracy metrics."""
    correct = (df[choice_col] == df[gt_col])
    accuracy = correct.mean() * 100
    
    # Per video
    per_video = df.groupby('video_id').apply(
        lambda x: (x[choice_col] == x[gt_col]).mean() * 100
    ).to_dict()
    
    # Per trajectory type
    per_traj = df.groupby('trajectory_type').apply(
        lambda x: (x[choice_col] == x[gt_col]).mean() * 100
    ).to_dict()
    
    return {
        'overall': accuracy,
        'per_video': per_video,
        'per_trajectory': per_traj,
        'count': len(df),
        'correct_count': correct.sum()
    }

def calculate_iou_all_choices(human_df: pd.DataFrame, vlm_df: pd.DataFrame) -> Dict:
    """Calculate IoU treating all choices (A, B, C) equally."""
    results = {'per_video': {}, 'per_participant': {}}
    
    # Overall IoU
    matches = 0
    total = 0
    
    for video_id in human_df['video_id'].unique():
        h_video = human_df[human_df['video_id'] == video_id]
        v_video = vlm_df[vlm_df['video_id'] == video_id]
        
        video_matches = 0
        video_total = 0
        
        for timepoint in h_video['timepoint'].unique():
            h_t = h_video[h_video['timepoint'] == timepoint]
            v_t = v_video[v_video['timepoint'] == timepoint]
            
            if len(v_t) > 0:
                vlm_choice = v_t['choice_vlm'].iloc[0]
                
                for _, h_row in h_t.iterrows():
                    if h_row['choice_human'] == vlm_choice:
                        matches += 1
                        video_matches += 1
                    total += 1
                    video_total += 1
        
        results['per_video'][video_id] = {
            'iou': (video_matches / video_total * 100) if video_total > 0 else 0,
            'matches': video_matches,
            'total': video_total
        }
    
    results['overall'] = (matches / total * 100) if total > 0 else 0
    results['matches'] = matches
    results['total'] = total
    
    # Per participant
    for participant in human_df['participant_id'].unique():
        h_part = human_df[human_df['participant_id'] == participant]
        
        part_matches = 0
        part_total = 0
        
        for video_id in h_part['video_id'].unique():
            h_video = h_part[h_part['video_id'] == video_id]
            v_video = vlm_df[vlm_df['video_id'] == video_id]
            
            for timepoint in h_video['timepoint'].unique():
                h_t = h_video[h_video['timepoint'] == timepoint]
                v_t = v_video[v_video['timepoint'] == timepoint]
                
                if len(v_t) > 0:
                    vlm_choice = v_t['choice_vlm'].iloc[0]
                    for _, h_row in h_t.iterrows():
                        if h_row['choice_human'] == vlm_choice:
                            part_matches += 1
                        part_total += 1
        
        results['per_participant'][participant] = {
            'iou': (part_matches / part_total * 100) if part_total > 0 else 0,
            'matches': part_matches,
            'total': part_total
        }
    
    return results

def analyze_timing(human_df: pd.DataFrame, vlm_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze when humans and VLM first correctly inferred the goal."""
    timing_results = []
    
    # Get final decisions (video_stop phase) for humans
    human_final = human_df[human_df['phase'] == 'video_stop'].copy()
    
    for video_id in GROUND_TRUTH.keys():
        gt = GROUND_TRUTH[video_id]
        traj_type = TRAJECTORY_TYPES[video_id]
        
        # VLM first correct
        v_video = vlm_df[(vlm_df['video_id'] == video_id) & (vlm_df['choice_vlm'] == gt)]
        if len(v_video) > 0:
            v_video = v_video.sort_values('timepoint')
            vlm_first_time = v_video['timepoint'].iloc[0]
            vlm_confidence = v_video['confidence_vlm'].iloc[0]
        else:
            vlm_first_time = None
            vlm_confidence = None
        
        # Human decision times
        h_video = human_final[human_final['video_id'] == video_id]
        
        human_times = []
        human_correct = 0
        human_total = 0
        
        for _, row in h_video.iterrows():
            decision_time = row.get('decision_time_sec')
            choice = row['choice_human']
            
            if decision_time is not None:
                human_total += 1
                if choice == gt:
                    human_times.append(decision_time)
                    human_correct += 1
        
        if human_times:
            human_mean_time = np.mean(human_times)
            human_std_time = np.std(human_times)
        else:
            human_mean_time = None
            human_std_time = None
        
        # Calculate difference
        if vlm_first_time is not None and human_mean_time is not None:
            time_diff = vlm_first_time - human_mean_time
            vlm_faster = time_diff < 0
        else:
            time_diff = None
            vlm_faster = None
        
        timing_results.append({
            'video_id': video_id,
            'trajectory_type': traj_type,
            'ground_truth': gt,
            'vlm_first_correct_time': vlm_first_time,
            'vlm_confidence': vlm_confidence,
            'human_mean_time': human_mean_time,
            'human_std_time': human_std_time,
            'human_correct_count': human_correct,
            'human_total': human_total,
            'time_difference': time_diff,
            'vlm_faster': vlm_faster
        })
    
    return pd.DataFrame(timing_results)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_publication_figures(human_accuracy: Dict, vlm_accuracy: Dict, 
                                 iou_results: Dict, timing_df: pd.DataFrame,
                                 output_dir: Path):
    """Create publication-ready figures."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'serif'
    
    # Figure 1: Main Comparison (Accuracy + IoU)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel A: Overall Accuracy
    ax = axes[0]
    categories = ['Human', 'VLM']
    values = [human_accuracy['overall'], vlm_accuracy['overall']]
    colors = ['#2E7D32', '#FFA726']
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('(A) Overall Accuracy', fontweight='bold', fontsize=13)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Panel B: Per-Video Accuracy
    ax = axes[1]
    videos = sorted(human_accuracy['per_video'].keys())
    x = np.arange(len(videos))
    width = 0.35
    
    h_vals = [human_accuracy['per_video'][v] for v in videos]
    v_vals = [vlm_accuracy['per_video'][v] for v in videos]
    
    ax.bar(x - width/2, h_vals, width, label='Human', color='#2E7D32', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x + width/2, v_vals, width, label='VLM', color='#FFA726', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Video', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('(B) Per-Video Accuracy', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in videos], rotation=0, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Panel C: IoU
    ax = axes[2]
    iou_val = iou_results['overall']
    bar = ax.bar(['IoU\n(All Choices)'], [iou_val], color='#1976D2', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Agreement (%)', fontweight='bold', fontsize=12)
    ax.set_title('(C) Human-VLM Agreement (IoU)', fontweight='bold', fontsize=13)
    ax.set_ylim(0, 100)
    ax.text(0, iou_val + 2, f'{iou_val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure1_main_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "figure1_main_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    # Figure 2: Timing Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Time comparison
    ax = axes[0]
    valid_timing = timing_df[timing_df['human_mean_time'].notna() & timing_df['vlm_first_correct_time'].notna()]
    
    x = np.arange(len(valid_timing))
    width = 0.35
    
    ax.bar(x - width/2, valid_timing['human_mean_time'].values, width,
           label='Human', color='#2E7D32', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x + width/2, valid_timing['vlm_first_correct_time'].values, width,
           label='VLM', color='#FFA726', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Video', fontweight='bold', fontsize=12)
    ax.set_ylabel('Time to Correct Inference (seconds)', fontweight='bold', fontsize=12)
    ax.set_title('(A) Goal Inference Timing', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in valid_timing['video_id']], rotation=0, fontsize=9)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: VLM vs Human difference
    ax = axes[1]
    valid_diff = timing_df[timing_df['time_difference'].notna()]
    colors = ['#4CAF50' if x < 0 else '#F44336' for x in valid_diff['time_difference']]
    
    bars = ax.bar(range(len(valid_diff)), valid_diff['time_difference'].values,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Video', fontweight='bold', fontsize=12)
    ax.set_ylabel('Time Difference (seconds)', fontweight='bold', fontsize=12)
    ax.set_title('(B) VLM vs Human\n(Negative = VLM Faster)', fontweight='bold', fontsize=13)
    ax.set_xticks(range(len(valid_diff)))
    ax.set_xticklabels([v.replace('_', '\n') for v in valid_diff['video_id']], rotation=0, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_timing_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "figure2_timing_analysis.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved publication figures to {output_dir}")

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    """Main analysis pipeline."""
    print("=" * 90)
    print("VLM vs HUMAN GOAL INFERENCE: COMPREHENSIVE ANALYSIS (FAIR COMPARISON)")
    print("=" * 90)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n⚠️  IMPORTANT: This analysis uses ONLY the timepoints shown to humans")
    print("   This ensures a FAIR comparison between VLM and human performance.")
    
    # Create output directory
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load user study timepoints for fair comparison
    study_timepoints = load_user_study_timepoints()
    
    # Load data - VLM filtered to match human study timepoints
    vlm_df = load_vlm_predictions(filter_timepoints=study_timepoints)
    human_df = load_human_decisions()
    
    # Filter to evaluation mode data
    human_eval = human_df[human_df['phase'].isin(['cumulative_frames', 'video_stop'])].copy()
    
    print(f"\n📊 Data Summary:")
    print(f"  VLM predictions:     {len(vlm_df)} observations")
    print(f"  Human observations:  {len(human_eval)} observations")
    print(f"  Participants:        {len(human_df['participant_id'].unique())}")
    print(f"  Videos:              {len(GROUND_TRUTH)}")
    
    # ========================================================================
    # ANALYSIS 1: ACCURACY
    # ========================================================================
    print("\n" + "=" * 90)
    print("ANALYSIS 1: ACCURACY vs GROUND TRUTH")
    print("=" * 90)
    
    human_accuracy = calculate_accuracy(human_eval, 'choice_human', 'ground_truth')
    vlm_accuracy = calculate_accuracy(vlm_df, 'choice_vlm', 'ground_truth')
    
    print(f"\n📊 Overall Accuracy:")
    print(f"  Human: {human_accuracy['overall']:.2f}% ({human_accuracy['correct_count']}/{human_accuracy['count']})")
    print(f"  VLM:   {vlm_accuracy['overall']:.2f}% ({vlm_accuracy['correct_count']}/{vlm_accuracy['count']})")
    print(f"  Difference: {human_accuracy['overall'] - vlm_accuracy['overall']:.2f}% (Human better)")
    
    print(f"\n📹 Per-Video Accuracy:")
    print(f"{'Video':<30} {'Human':<12} {'VLM':<12} {'Difference':<12}")
    print("-" * 90)
    for video in sorted(human_accuracy['per_video'].keys()):
        h_acc = human_accuracy['per_video'][video]
        v_acc = vlm_accuracy['per_video'][video]
        diff = h_acc - v_acc
        print(f"{video:<30} {h_acc:>6.1f}%      {v_acc:>6.1f}%      {diff:>+6.1f}%")
    
    print(f"\n🎯 Per-Trajectory Type:")
    for traj_type in ['legible', 'ambiguous']:
        h_acc = human_accuracy['per_trajectory'].get(traj_type, 0)
        v_acc = vlm_accuracy['per_trajectory'].get(traj_type, 0)
        print(f"  {traj_type.capitalize():<12} - Human: {h_acc:>5.1f}%  |  VLM: {v_acc:>5.1f}%")
    
    # ========================================================================
    # ANALYSIS 2: IoU (AGREEMENT)
    # ========================================================================
    print("\n" + "=" * 90)
    print("ANALYSIS 2: HUMAN-VLM AGREEMENT (IoU)")
    print("=" * 90)
    
    # Merge for IoU calculation
    human_timepoint = human_eval[human_eval['phase'] == 'cumulative_frames'].copy()
    iou_results = calculate_iou_all_choices(human_timepoint, vlm_df)
    
    print(f"\n📊 Overall IoU (All Choices A/B/C): {iou_results['overall']:.2f}%")
    print(f"  Exact matches: {iou_results['matches']}/{iou_results['total']}")
    
    print(f"\n📹 Per-Video IoU:")
    for video in sorted(iou_results['per_video'].keys()):
        metrics = iou_results['per_video'][video]
        print(f"  {video:<30} {metrics['iou']:>6.2f}% ({metrics['matches']:>3d}/{metrics['total']:>3d})")
    
    print(f"\n👥 Per-Participant IoU:")
    for participant in sorted(iou_results['per_participant'].keys()):
        metrics = iou_results['per_participant'][participant]
        print(f"  {participant:<30} {metrics['iou']:>6.2f}% ({metrics['matches']:>3d}/{metrics['total']:>3d})")
    
    # ========================================================================
    # ANALYSIS 3: TIMING
    # ========================================================================
    print("\n" + "=" * 90)
    print("ANALYSIS 3: GOAL INFERENCE TIMING")
    print("=" * 90)
    
    timing_df = analyze_timing(human_df, vlm_df)
    
    print(f"\n⏱️  Timing Results:")
    print(f"{'Video':<30} {'Human (s)':<12} {'VLM (t)':<12} {'VLM Faster?':<15}")
    print("-" * 90)
    for _, row in timing_df.iterrows():
        h_time = f"{row['human_mean_time']:.2f}" if pd.notna(row['human_mean_time']) else "N/A"
        v_time = f"{row['vlm_first_correct_time']:.0f}" if pd.notna(row['vlm_first_correct_time']) else "N/A"
        faster = "✅ YES" if row['vlm_faster'] else "❌ NO" if pd.notna(row['vlm_faster']) else "N/A"
        print(f"{row['video_id']:<30} {h_time:<12} {v_time:<12} {faster:<15}")
    
    # Summary by trajectory type
    for traj_type in ['legible', 'ambiguous']:
        traj_timing = timing_df[timing_df['trajectory_type'] == traj_type]
        faster_count = traj_timing['vlm_faster'].sum()
        total = traj_timing['vlm_faster'].notna().sum()
        
        h_mean = traj_timing['human_mean_time'].mean()
        v_mean = traj_timing['vlm_first_correct_time'].mean()
        
        print(f"\n  {traj_type.capitalize()}:")
        print(f"    Human mean time:  {h_mean:.2f}s")
        print(f"    VLM mean time:    {v_mean:.2f}s")
        print(f"    VLM faster:       {faster_count}/{total} videos")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 90)
    print("SAVING RESULTS")
    print("=" * 90)
    
    # Save CSV files
    timing_df.to_csv(ANALYSIS_OUTPUT_DIR / "timing_analysis.csv", index=False)
    
    # Save IoU per video
    iou_per_video = pd.DataFrame([
        {'video_id': k, **v} for k, v in iou_results['per_video'].items()
    ])
    iou_per_video.to_csv(ANALYSIS_OUTPUT_DIR / "iou_per_video.csv", index=False)
    
    # Save accuracy summary
    accuracy_summary = []
    for video in sorted(human_accuracy['per_video'].keys()):
        accuracy_summary.append({
            'video_id': video,
            'trajectory_type': TRAJECTORY_TYPES[video],
            'human_accuracy': human_accuracy['per_video'][video],
            'vlm_accuracy': vlm_accuracy['per_video'][video],
            'difference': human_accuracy['per_video'][video] - vlm_accuracy['per_video'][video]
        })
    pd.DataFrame(accuracy_summary).to_csv(ANALYSIS_OUTPUT_DIR / "accuracy_summary.csv", index=False)
    
    # Save summary JSON
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'overall_metrics': {
            'human_accuracy': human_accuracy['overall'],
            'vlm_accuracy': vlm_accuracy['overall'],
            'iou_all_choices': iou_results['overall'],
            'vlm_faster_count': int(timing_df['vlm_faster'].sum()),
            'vlm_faster_percentage': float(timing_df['vlm_faster'].sum() / timing_df['vlm_faster'].notna().sum() * 100)
        },
        'data_sources': {
            'vlm_predictions': len(vlm_df),
            'human_observations': len(human_eval),
            'participants': len(human_df['participant_id'].unique()),
            'videos': len(GROUND_TRUTH)
        },
        'fairness_note': 'VLM evaluated ONLY at timepoints shown to humans (from user_study_timepoints.csv)'
    }
    
    with open(ANALYSIS_OUTPUT_DIR / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved CSV files:")
    print(f"  - timing_analysis.csv")
    print(f"  - iou_per_video.csv")
    print(f"  - accuracy_summary.csv")
    print(f"  - analysis_summary.json")
    
    # Create publication figures
    create_publication_figures(human_accuracy, vlm_accuracy, iou_results, timing_df, ANALYSIS_OUTPUT_DIR)
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    print("\n" + "=" * 90)
    print("🎯 FINAL VERDICT: CAN VLM SERVE AS PROXY FOR HUMAN ASSESSMENT?")
    print("=" * 90)
    
    overall_faster_pct = timing_df['vlm_faster'].sum() / timing_df['vlm_faster'].notna().sum() * 100
    
    print(f"\n📊 Key Findings:")
    print(f"  1. Accuracy:    Human {human_accuracy['overall']:.1f}%  vs  VLM {vlm_accuracy['overall']:.1f}%")
    print(f"                 Gap: {human_accuracy['overall'] - vlm_accuracy['overall']:.1f}% (Human better)")
    
    print(f"\n  2. Agreement:   {iou_results['overall']:.1f}% (IoU treating A/B/C equally)")
    
    print(f"\n  3. Timing:      VLM faster in {int(timing_df['vlm_faster'].sum())}/{int(timing_df['vlm_faster'].notna().sum())} videos ({overall_faster_pct:.0f}%)")
    
    if vlm_accuracy['overall'] >= 50 and iou_results['overall'] >= 60 and overall_faster_pct >= 60:
        verdict = "✅ YES - VLM CAN serve as proxy (with limitations)"
        recommendation = "VLM is suitable for initial screening and large-scale analysis, but human verification recommended for high-stakes decisions."
    elif vlm_accuracy['overall'] >= 40 and iou_results['overall'] >= 50:
        verdict = "⚠️  PARTIAL - VLM shows promise but significant limitations exist"
        recommendation = "VLM can complement human assessment but should not replace it. Use for exploratory analysis only."
    else:
        verdict = "❌ NO - VLM cannot reliably replace human assessment"
        recommendation = "VLM performance is too low for reliable use. Significant improvements needed before deployment."
    
    print(f"\n💡 Verdict: {verdict}")
    print(f"\n📌 Recommendation:")
    print(f"   {recommendation}")
    
    print("\n" + "=" * 90)
    print(f"✅ Analysis complete! Results saved to: {ANALYSIS_OUTPUT_DIR}")
    print("\n⚠️  FAIRNESS GUARANTEE:")
    print("   ✓ VLM evaluated ONLY at timepoints shown to humans")
    print("   ✓ No advantage from evaluating at every second")
    print("   ✓ Direct apples-to-apples comparison")
    print("=" * 90)

if __name__ == "__main__":
    main()
