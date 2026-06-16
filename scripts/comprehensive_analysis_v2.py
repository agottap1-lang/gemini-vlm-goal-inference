#!/usr/bin/env python3
"""
VLM vs Human Goal Inference: Comprehensive Analysis V2
========================================================

Updated analysis using ALL 8 participants with focus on accuracy and IoU metrics.

Author: Research Analysis Pipeline V2
Date: April 2026
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
PARTICIPANT_DATA_DIR = Path(r"C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\participant_data")
VLM_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
ANALYSIS_OUTPUT_DIR = Path(__file__).parent.parent / "analysis_results_2"
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

# Participant anonymization map (sorted alphabetically; non-ASCII → P8)
ANON_MAP = {
    "Abhi": "P1", "Kartikay": "P2", "Prabhath": "P3", "Raj": "P4",
    "Ryan": "P5", "Sho": "P6", "Summu": "P7",
}
def _anon(name: str) -> str:
    if isinstance(name, str) and name.isascii():
        return ANON_MAP.get(name, name)
    return "P8"

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
    """Load VLM predictions from the LATEST run directories."""
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
    """Load human participant decisions from ALL participants."""
    print("\n👥 Loading Human Decisions (ALL PARTICIPANTS)...")
    
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
            print(f"  ✓ Loaded {json_file.name}")
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

    # Anonymize participant IDs
    df['participant_id'] = df['participant_id'].apply(_anon)

    print(f"\n✓ Loaded {len(df)} observations from {len(df['participant_id'].unique())} participants")
    print(f"  Participants: {sorted(df['participant_id'].unique())}")
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

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_publication_figures(human_accuracy: Dict, vlm_accuracy: Dict,
                                 iou_results: Dict, output_dir: Path):
    """Create clean publication-ready figures: 4 figures total."""
    C_LEG   = '#1B6CA8'   # blue  – legible trajectories
    C_AMB   = '#C0392B'   # red   – ambiguous trajectories
    C_HUMAN = '#1B4332'   # dark green – human observer
    C_VLM   = '#B45309'   # amber – VLM observer
    C_GRID  = '#E5E7EB'

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
    })

    # Canonical full video name labels
    VID_LABEL = {
        'amb_d_drawer_close':  'Amb: Bottom Drawer',
        'amb_l_block':         'Amb: Left Block',
        'amb_r_block':         'Amb: Right Block',
        'amb_to_drawer_close': 'Amb: Top Drawer',
        'le_d_drawer_close':   'Leg: Bottom Drawer',
        'le_l_block':          'Leg: Left Block',
        'le_r_block':          'Leg: Right Block',
        'le_t_drawer_close':   'Leg: Top Drawer',
    }
    # Short x-axis labels (trajectory type prefix on first line)
    VID_XLAB = {
        'amb_d_drawer_close':  'Bottom\nDrawer',
        'amb_l_block':         'Left\nBlock',
        'amb_r_block':         'Right\nBlock',
        'amb_to_drawer_close': 'Top\nDrawer',
        'le_d_drawer_close':   'Bottom\nDrawer',
        'le_l_block':          'Left\nBlock',
        'le_r_block':          'Right\nBlock',
        'le_t_drawer_close':   'Top\nDrawer',
    }

    leg_vids = sorted([v for v in human_accuracy['per_video'] if TRAJECTORY_TYPES[v] == 'legible'])
    amb_vids = sorted([v for v in human_accuracy['per_video'] if TRAJECTORY_TYPES[v] == 'ambiguous'])
    all_vids = leg_vids + amb_vids   # Legible first, then Ambiguous

    # ─── Figure 1: Grouped bar chart — Human vs VLM accuracy per video ───────
    x = np.arange(len(all_vids))
    w = 0.38
    h_vals = [human_accuracy['per_video'][v] for v in all_vids]
    v_vals = [vlm_accuracy['per_video'][v]   for v in all_vids]
    h_mean = human_accuracy['overall']
    v_mean = vlm_accuracy['overall']

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars_h = ax.bar(x - w / 2, h_vals, width=w, color=C_HUMAN, alpha=0.85,
                    label=f'Human (mean {h_mean:.1f}%)', zorder=3)
    bars_v = ax.bar(x + w / 2, v_vals, width=w, color=C_VLM, alpha=0.85,
                    label=f'VLM – Gemini (mean {v_mean:.1f}%)', zorder=3)
    # Value labels above each bar
    for bar in bars_h:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{bar.get_height():.0f}%', ha='center', va='bottom',
                fontsize=8, color=C_HUMAN, fontweight='bold')
    for bar in bars_v:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{bar.get_height():.0f}%', ha='center', va='bottom',
                fontsize=8, color=C_VLM, fontweight='bold')
    # Chance line
    ax.axhline(50, color='grey', lw=1.0, ls=':', alpha=0.6, zorder=1)
    ax.text(0.1, 51.5, 'Chance (50%)',
            ha='left', fontsize=8, color='grey', style='italic')
    # Group background shading + section labels
    ax.axvspan(-0.55, len(leg_vids) - 0.5, alpha=0.05, color=C_LEG, zorder=0)
    ax.axvspan(len(leg_vids) - 0.5, len(all_vids) - 0.45, alpha=0.05, color=C_AMB, zorder=0)
    ax.text(len(leg_vids) / 2 - 0.5, 92,
            'Legible Trajectories', ha='center', fontsize=9,
            color=C_LEG, fontweight='bold')
    ax.text(len(leg_vids) + len(amb_vids) / 2 - 0.5, 92,
            'Ambiguous Trajectories', ha='center', fontsize=9,
            color=C_AMB, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([VID_XLAB[v] for v in all_vids], fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 0.99),
              ncol=2, framealpha=0.9)
    ax.set_title('Goal Inference Accuracy: Human vs VLM (Gemini) per Video',
                 fontsize=13, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(output_dir / 'figure1_main_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure1_main_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ Saved: figure1_main_comparison.png/pdf')

    # ─── Figure 2: Per-Participant Choice Agreement — horizontal bar chart ────
    participants = sorted(iou_results['per_participant'].keys())
    p_iou       = [iou_results['per_participant'][p]['iou'] for p in participants]
    overall_iou = iou_results['overall']
    sorted_pairs = sorted(zip(participants, p_iou), key=lambda x: x[1])
    s_parts, s_iou = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(s_parts))
    ax.barh(y, list(s_iou), color='#1D4ED8', alpha=0.82, height=0.6, zorder=3)
    ax.axvline(overall_iou, color='#DC2626', lw=1.8, ls='--', zorder=4, alpha=0.9,
               label=f'Mean = {overall_iou:.1f}%')
    for xi, yi in zip(s_iou, y):
        ax.text(xi + 0.5, yi, f'{xi:.1f}%', va='center', ha='left',
                fontsize=10, color='#1F2937', fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(list(s_parts), fontsize=11)
    ax.set_xlabel('Choice Agreement with VLM (%)', fontsize=12)
    ax.set_xlim(55, 93)
    ax.set_title('Per-Participant Choice Agreement with VLM\n'
                 '(% of timepoints where choice matched VLM prediction)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax.xaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / 'figure2_participant_iou.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure2_participant_iou.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ Saved: figure2_participant_iou.png/pdf')

    # ─── Figure 3: Per-Video Choice Agreement — horizontal bar chart ──────────
    vid_iou_pairs = sorted(
        [(v, iou_results['per_video'][v]['iou']) for v in iou_results['per_video']],
        key=lambda x: x[1]
    )  # ascending → highest bar at top
    s_vids, s_vio = zip(*vid_iou_pairs)

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(s_vids))
    for yi, v, val in zip(y, s_vids, s_vio):
        col = C_AMB if TRAJECTORY_TYPES[v] == 'ambiguous' else C_LEG
        ax.barh(yi, val, color=col, alpha=0.82, height=0.6, zorder=3)
    ax.axvline(iou_results['overall'], color='#374151', lw=1.8, ls='--',
               alpha=0.8, zorder=4)
    # Mean label placed safely inside the right margin (below top bar)
    ax.text(iou_results['overall'] + 0.8, len(s_vids) - 1.5,
            f"Mean = {iou_results['overall']:.1f}%",
            fontsize=9, color='#374151', fontweight='bold', va='center')
    for xi, yi, v in zip(s_vio, y, s_vids):
        star = ' ★' if v == 'amb_r_block' else ''
        ax.text(xi + 0.8, yi, f'{xi:.1f}%{star}', va='center', ha='left',
                fontsize=9.5, color='#1F2937', fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels([VID_LABEL[v] for v in s_vids], fontsize=10)
    ax.set_xlabel('Choice Agreement (%)', fontsize=12)
    ax.set_xlim(0, 118)
    ax.set_title('Per-Video Human–VLM Choice Agreement\n'
                 '(% of timepoints where participant matched VLM prediction; ★ = outlier)',
                 fontsize=12, fontweight='bold')
    ax.legend(handles=[
        mpatches.Patch(color=C_LEG, label='Legible trajectory'),
        mpatches.Patch(color=C_AMB, label='Ambiguous trajectory'),
    ], fontsize=10, loc='lower right', framealpha=0.9)
    ax.xaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / 'figure3_video_iou.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure3_video_iou.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ Saved: figure3_video_iou.png/pdf')

    # ─── Figure 4: Summary table — accuracy & agreement per video ─────────────
    col_labels = ['Video', 'Type', 'Human Acc.', 'VLM Acc.', 'Agreement']
    table_rows = []
    for v in leg_vids + amb_vids:
        ttype = 'Legible' if TRAJECTORY_TYPES[v] == 'legible' else 'Ambiguous'
        name  = VID_LABEL[v].split(': ')[1]
        h_a   = human_accuracy['per_video'][v]
        v_a   = vlm_accuracy['per_video'][v]
        agr   = iou_results['per_video'][v]['iou']
        table_rows.append([name, ttype, f'{h_a:.1f}%', f'{v_a:.1f}%', f'{agr:.1f}%'])
    # Summary row
    table_rows.append([
        'Mean', '—',
        f"{human_accuracy['overall']:.1f}%",
        f"{vlm_accuracy['overall']:.1f}%",
        f"{iou_results['overall']:.1f}%",
    ])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')
    tbl = ax.table(cellText=table_rows, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    # Header row styling
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#1F2937')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    # Data row styling (alternating tint by type)
    for i in range(1, len(table_rows) + 1):
        if i <= len(leg_vids):
            bg = '#EFF6FF'   # light blue – Legible
        elif i <= len(leg_vids) + len(amb_vids):
            bg = '#FEF2F2'   # light red – Ambiguous
        else:
            bg = '#F3F4F6'   # grey – Mean row
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor(bg)
    # Bold mean row
    for j in range(len(col_labels)):
        tbl[(len(table_rows), j)].set_text_props(fontweight='bold')
        tbl[(len(table_rows), j)].set_facecolor('#F3F4F6')
    ax.set_title(
        'Summary: Goal Inference Accuracy and Human–VLM Agreement per Video\n'
        '(blue rows = Legible trajectories; red rows = Ambiguous trajectories)',
        fontsize=11, fontweight='bold', pad=14)
    fig.tight_layout()
    fig.savefig(output_dir / 'figure4_summary_table.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure4_summary_table.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ Saved: figure4_summary_table.png/pdf')

    print(f'\n✓ All figures saved to {output_dir}')

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    """Main analysis pipeline - Version 2 with all participants."""
    print("=" * 90)
    print("VLM vs HUMAN GOAL INFERENCE: COMPREHENSIVE ANALYSIS V2 (ALL PARTICIPANTS)")
    print("=" * 90)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n⚠️  UPDATES IN V2:")
    print("   - Using ALL 8 participants (previous analysis used 6)")
    print("   - Focus on Accuracy and IoU metrics")
    print("   - VLM evaluated ONLY at timepoints shown to humans")
    
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
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 90)
    print("SAVING RESULTS")
    print("=" * 90)
    
    # Save accuracy summary
    accuracy_data = []
    for video in sorted(human_accuracy['per_video'].keys()):
        accuracy_data.append({
            'video_id': video,
            'trajectory_type': TRAJECTORY_TYPES[video],
            'human_accuracy': human_accuracy['per_video'][video],
            'vlm_accuracy': vlm_accuracy['per_video'][video],
            'difference': human_accuracy['per_video'][video] - vlm_accuracy['per_video'][video]
        })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df.to_csv(ANALYSIS_OUTPUT_DIR / "accuracy_summary.csv", index=False)
    print(f"  ✓ Saved: accuracy_summary.csv")
    
    # Save IoU per video
    iou_video_data = []
    for video in sorted(iou_results['per_video'].keys()):
        metrics = iou_results['per_video'][video]
        iou_video_data.append({
            'video_id': video,
            'trajectory_type': TRAJECTORY_TYPES[video],
            'iou': metrics['iou'],
            'matches': metrics['matches'],
            'total': metrics['total']
        })
    
    iou_video_df = pd.DataFrame(iou_video_data)
    iou_video_df.to_csv(ANALYSIS_OUTPUT_DIR / "iou_per_video.csv", index=False)
    print(f"  ✓ Saved: iou_per_video.csv")
    
    # Save IoU per participant
    iou_participant_data = []
    for participant in sorted(iou_results['per_participant'].keys()):
        metrics = iou_results['per_participant'][participant]
        iou_participant_data.append({
            'participant_id': participant,
            'iou': metrics['iou'],
            'matches': metrics['matches'],
            'total': metrics['total']
        })
    
    iou_participant_df = pd.DataFrame(iou_participant_data)
    iou_participant_df.to_csv(ANALYSIS_OUTPUT_DIR / "iou_per_participant.csv", index=False)
    print(f"  ✓ Saved: iou_per_participant.csv")
    
    # Save summary JSON
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'version': 2,
        'overall_metrics': {
            'human_accuracy': human_accuracy['overall'],
            'vlm_accuracy': vlm_accuracy['overall'],
            'iou_all_choices': iou_results['overall']
        },
        'data_sources': {
            'vlm_predictions': len(vlm_df),
            'human_observations': len(human_eval),
            'participants': len(human_df['participant_id'].unique()),
            'videos': len(GROUND_TRUTH)
        },
        'participants_list': sorted(human_df['participant_id'].unique()),
        'fairness_note': 'VLM evaluated ONLY at timepoints shown to humans (from user_study_timepoints.csv)'
    }
    
    with open(ANALYSIS_OUTPUT_DIR / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: analysis_summary.json")
    
    # ========================================================================
    # CREATE FIGURES
    # ========================================================================
    print("\n" + "=" * 90)
    print("CREATING PUBLICATION FIGURES")
    print("=" * 90)
    
    create_publication_figures(human_accuracy, vlm_accuracy, iou_results, ANALYSIS_OUTPUT_DIR)
    
    print("\n" + "=" * 90)
    print("✅ ANALYSIS V2 COMPLETE!")
    print("=" * 90)
    print(f"\nAll results saved to: {ANALYSIS_OUTPUT_DIR}")
    print("\nKey Findings:")
    print(f"  • {len(human_df['participant_id'].unique())} participants analyzed")
    print(f"  • Human accuracy: {human_accuracy['overall']:.1f}%")
    print(f"  • VLM accuracy: {vlm_accuracy['overall']:.1f}%")
    print(f"  • Human-VLM agreement (IoU): {iou_results['overall']:.1f}%")

if __name__ == "__main__":
    main()
