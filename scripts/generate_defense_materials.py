#!/usr/bin/env python3
"""
Generate Additional Defense Materials
Statistical tests, confusion matrices, error analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import binomtest, chi2_contingency, fisher_exact
from scipy import stats

# Paths
ANALYSIS_DIR = Path(__file__).parent.parent / "analysis_results_2"
OUTPUT_DIR = ANALYSIS_DIR / "defense_materials"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
PARTICIPANT_DATA_DIR = Path(r"C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\participant_data")
VLM_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
USER_STUDY_TIMEPOINTS_FILE = Path(__file__).parent.parent / "user_study_timepoints.csv"

VIDEO_ID_MAPPING = {"amb_d_drawer_cclose": "amb_d_drawer_close"}

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
    """Load timepoints."""
    df = pd.read_csv(USER_STUDY_TIMEPOINTS_FILE)
    timepoints_map = {}
    for _, row in df.iterrows():
        video_id = row['video_id']
        timepoints_str = row['recommended_timepoints']
        timepoints = [int(t.strip()) for t in timepoints_str.split(',')]
        timepoints_map[video_id] = timepoints
    return timepoints_map

def find_latest_vlm_run(video_id: str) -> Path:
    """Find the most recent VLM run directory."""
    video_dir = VLM_OUTPUTS_DIR / video_id
    run_dirs = sorted([d for d in video_dir.glob("run_*_prefix") if d.is_dir()])
    latest_run = run_dirs[-1]
    return latest_run / "results.jsonl"

def load_vlm_predictions(filter_timepoints):
    """Load VLM predictions."""
    all_data = []
    for video_id in GROUND_TRUTH.keys():
        try:
            latest_file = find_latest_vlm_run(video_id)
            with open(latest_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        pred['video_id'] = video_id
                        pred['ground_truth'] = GROUND_TRUTH[video_id]
                        all_data.append(pred)
        except Exception as e:
            print(f"Error loading {video_id}: {e}")
    
    df = pd.DataFrame(all_data)
    df = df.rename(columns={'choice': 'choice_vlm', 't_sec': 'timepoint'})
    df['choice_vlm'] = df['choice_vlm'].str.upper()
    df['timepoint'] = pd.to_numeric(df['timepoint'], errors='coerce').fillna(0).astype(int)
    
    # Filter to timepoints
    df = df[df.apply(lambda row: row['timepoint'] in filter_timepoints.get(row['video_id'], []), axis=1)]
    return df

def load_human_decisions():
    """Load human decisions."""
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
                        if video_id in GROUND_TRUTH:
                            obs['video_id'] = video_id
                            obs['ground_truth'] = GROUND_TRUTH.get(video_id)
                            all_data.append(obs)
        except Exception as e:
            print(f"Error: {e}")
    
    df = pd.DataFrame(all_data)
    df = df.rename(columns={'choice': 'choice_human', 'confidence_0_10': 'confidence_human'})
    df['choice_human'] = df['choice_human'].str.upper()
    df['timepoint'] = pd.to_numeric(df['timepoint'], errors='coerce').fillna(0).astype(int)
    return df

def create_confusion_matrices(human_df, vlm_df):
    """Create confusion matrices for humans and VLM."""
    print("\n" + "="*80)
    print("CONFUSION MATRICES")
    print("="*80)
    
    choices = ['A', 'B', 'C']
    
    # Human confusion matrix
    human_eval = human_df[human_df['phase'].isin(['cumulative_frames', 'video_stop'])].copy()
    human_cm = pd.crosstab(
        human_eval['ground_truth'], 
        human_eval['choice_human'],
        rownames=['True'],
        colnames=['Predicted'],
        margins=True
    )
    
    # VLM confusion matrix
    vlm_cm = pd.crosstab(
        vlm_df['ground_truth'],
        vlm_df['choice_vlm'],
        rownames=['True'],
        colnames=['Predicted'],
        margins=True
    )
    
    print("\nHuman Confusion Matrix:")
    print(human_cm)
    print("\nVLM Confusion Matrix:")
    print(vlm_cm)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Human
    ax = axes[0]
    human_cm_norm = pd.crosstab(
        human_eval['ground_truth'], 
        human_eval['choice_human'],
        normalize='index'
    ) * 100
    sns.heatmap(human_cm_norm, annot=True, fmt='.1f', cmap='Blues', ax=ax, 
                vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'})
    ax.set_title('Human Confusion Matrix\n(% of each true class)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Predicted Goal', fontweight='bold')
    ax.set_ylabel('True Goal', fontweight='bold')
    
    # VLM
    ax = axes[1]
    vlm_cm_norm = pd.crosstab(
        vlm_df['ground_truth'],
        vlm_df['choice_vlm'],
        normalize='index'
    ) * 100
    sns.heatmap(vlm_cm_norm, annot=True, fmt='.1f', cmap='Oranges', ax=ax,
                vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'})
    ax.set_title('VLM Confusion Matrix\n(% of each true class)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Predicted Goal', fontweight='bold')
    ax.set_ylabel('True Goal', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "confusion_matrices.pdf", bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved confusion matrices")
    
    return human_cm, vlm_cm

def statistical_tests(human_df, vlm_df):
    """Perform statistical significance tests."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    results = {}
    
    # Filter to evaluation data
    human_eval = human_df[human_df['phase'].isin(['cumulative_frames', 'video_stop'])].copy()
    
    # Overall accuracy tests
    human_correct = (human_eval['choice_human'] == human_eval['ground_truth']).sum()
    human_total = len(human_eval)
    
    vlm_correct = (vlm_df['choice_vlm'] == vlm_df['ground_truth']).sum()
    vlm_total = len(vlm_df)
    
    # Test 1: Human performance > chance (33.3%)
    human_binom = binomtest(human_correct, human_total, 1/3, alternative='greater')
    results['human_vs_chance'] = {
        'accuracy': human_correct / human_total * 100,
        'p_value': human_binom.pvalue,
        'significant': human_binom.pvalue < 0.05
    }
    print(f"\n1. Human Accuracy vs Chance (33.3%)")
    print(f"   Accuracy: {human_correct}/{human_total} = {human_correct/human_total*100:.2f}%")
    print(f"   p-value: {human_binom.pvalue:.2e}")
    print(f"   Result: {'✓ Significantly above chance' if human_binom.pvalue < 0.05 else 'Not significant'}")
    
    # Test 2: VLM performance > chance
    vlm_binom = binomtest(vlm_correct, vlm_total, 1/3, alternative='greater')
    results['vlm_vs_chance'] = {
        'accuracy': vlm_correct / vlm_total * 100,
        'p_value': vlm_binom.pvalue,
        'significant': vlm_binom.pvalue < 0.05
    }
    print(f"\n2. VLM Accuracy vs Chance (33.3%)")
    print(f"   Accuracy: {vlm_correct}/{vlm_total} = {vlm_correct/vlm_total*100:.2f}%")
    print(f"   p-value: {vlm_binom.pvalue:.2e}")
    print(f"   Result: {'✓ Significantly above chance' if vlm_binom.pvalue < 0.05 else 'Not significant'}")
    
    # Test 3: Human vs VLM (Chi-square on contingency table)
    contingency = np.array([
        [human_correct, human_total - human_correct],
        [vlm_correct, vlm_total - vlm_correct]
    ])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    results['human_vs_vlm_overall'] = {
        'chi2': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    print(f"\n3. Human vs VLM Overall Accuracy")
    print(f"   Human: {human_correct/human_total*100:.2f}%")
    print(f"   VLM:   {vlm_correct/vlm_total*100:.2f}%")
    print(f"   Difference: {abs(human_correct/human_total - vlm_correct/vlm_total)*100:.2f}%")
    print(f"   Chi-square: {chi2:.4f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Result: {'Significantly different' if p_value < 0.05 else '✓ Not significantly different (comparable performance)'}")
    
    # Test 4: Per-video comparisons
    print(f"\n4. Per-Video Human vs VLM (Fisher's Exact Test)")
    print(f"   {'Video':<25} {'Human %':<10} {'VLM %':<10} {'p-value':<12} {'Significant?':<15}")
    print("   " + "-"*75)
    
    video_tests = {}
    for video_id in GROUND_TRUTH.keys():
        h_video = human_eval[human_eval['video_id'] == video_id]
        v_video = vlm_df[vlm_df['video_id'] == video_id]
        
        h_correct = (h_video['choice_human'] == h_video['ground_truth']).sum()
        h_total = len(h_video)
        v_correct = (v_video['choice_vlm'] == v_video['ground_truth']).sum()
        v_total = len(v_video)
        
        # Fisher's exact test (better for small samples)
        contingency_video = np.array([
            [h_correct, h_total - h_correct],
            [v_correct, v_total - v_correct]
        ])
        
        if contingency_video.sum() > 0:
            odds_ratio, p_val = fisher_exact(contingency_video)
            video_tests[video_id] = {
                'human_acc': h_correct/h_total*100,
                'vlm_acc': v_correct/v_total*100,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            
            sig_marker = "✓ YES" if p_val < 0.05 else "No"
            print(f"   {video_id:<25} {h_correct/h_total*100:>6.1f}%    {v_correct/v_total*100:>6.1f}%    {p_val:>8.4f}    {sig_marker:<15}")
    
    results['per_video_tests'] = video_tests
    
    # Save results (convert numpy bools to python bools for JSON)
    def convert_to_serializable(obj):
        if isinstance(obj, (np.bool_, np.ndarray)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(OUTPUT_DIR / "statistical_tests.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Statistical tests saved to {OUTPUT_DIR / 'statistical_tests.json'}")
    
    return results

def error_analysis(human_df, vlm_df):
    """Analyze what kinds of errors are made."""
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    human_eval = human_df[human_df['phase'].isin(['cumulative_frames', 'video_stop'])].copy()
    
    # Get errors
    human_errors = human_eval[human_eval['choice_human'] != human_eval['ground_truth']].copy()
    vlm_errors = vlm_df[vlm_df['choice_vlm'] != vlm_df['ground_truth']].copy()
    
    print(f"\nTotal errors:")
    print(f"  Human: {len(human_errors)}/{len(human_eval)} = {len(human_errors)/len(human_eval)*100:.1f}%")
    print(f"  VLM:   {len(vlm_errors)}/{len(vlm_df)} = {len(vlm_errors)/len(vlm_df)*100:.1f}%")
    
    # Error types (which goal was confused)
    print(f"\n{'Error Type':<30} {'Human Count':<15} {'VLM Count':<15}")
    print("-"*80)
    
    error_types = {}
    for true_goal in ['A', 'B']:
        for pred_goal in ['A', 'B', 'C']:
            if true_goal != pred_goal:
                error_type = f"{true_goal} → {pred_goal}"
                h_count = len(human_errors[
                    (human_errors['ground_truth'] == true_goal) & 
                    (human_errors['choice_human'] == pred_goal)
                ])
                v_count = len(vlm_errors[
                    (vlm_errors['ground_truth'] == true_goal) & 
                    (vlm_errors['choice_vlm'] == pred_goal)
                ])
                error_types[error_type] = {'human': h_count, 'vlm': v_count}
                print(f"{error_type:<30} {h_count:<15} {v_count:<15}")
    
    # Shared errors (both wrong on same observation)
    print(f"\n{'Video':<30} {'Shared Errors':<20} {'Total Observations':<20}")
    print("-"*80)
    
    for video_id in GROUND_TRUTH.keys():
        h_video = human_eval[human_eval['video_id'] == video_id]
        v_video = vlm_df[vlm_df['video_id'] == video_id]
        
        # For each timepoint, check if both made errors
        shared_errors = 0
        total_comparisons = 0
        
        for timepoint in v_video['timepoint'].unique():
            v_t = v_video[v_video['timepoint'] == timepoint]
            h_t = h_video[h_video['timepoint'] == timepoint]
            
            if len(v_t) > 0 and len(h_t) > 0:
                vlm_choice = v_t['choice_vlm'].iloc[0]
                gt = v_t['ground_truth'].iloc[0]
                vlm_wrong = (vlm_choice != gt)
                
                for _, h_row in h_t.iterrows():
                    total_comparisons += 1
                    if vlm_wrong and (h_row['choice_human'] != gt):
                        shared_errors += 1
        
        if total_comparisons > 0:
            print(f"{video_id:<30} {shared_errors:<20} {total_comparisons:<20}")
    
    # Visualize error patterns
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    error_df = pd.DataFrame(error_types).T
    error_df.plot(kind='bar', ax=ax, color=['#2E7D32', '#FFA726'], 
                   edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Error Type (True → Predicted)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Errors', fontweight='bold', fontsize=12)
    ax.set_title('Error Type Distribution: Human vs VLM', fontweight='bold', fontsize=13)
    ax.legend(['Human', 'VLM'], frameon=True, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "error_analysis.pdf", bbox_inches='tight')
    plt.close()
    print(f"\n✓ Error analysis visualization saved")
    
    return error_types

def create_scatter_comparison(human_df, vlm_df):
    """Create scatterplot comparing human vs VLM accuracy per video."""
    print("\n" + "="*80)
    print("CREATING SCATTER COMPARISON")
    print("="*80)
    
    human_eval = human_df[human_df['phase'].isin(['cumulative_frames', 'video_stop'])].copy()
    
    video_accuracies = []
    for video_id in GROUND_TRUTH.keys():
        h_video = human_eval[human_eval['video_id'] == video_id]
        v_video = vlm_df[vlm_df['video_id'] == video_id]
        
        h_acc = (h_video['choice_human'] == h_video['ground_truth']).mean() * 100
        v_acc = (v_video['choice_vlm'] == v_video['ground_truth']).mean() * 100
        
        video_accuracies.append({
            'video_id': video_id,
            'human_acc': h_acc,
            'vlm_acc': v_acc
        })
    
    df = pd.DataFrame(video_accuracies)
    
    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    colors = ['#e74c3c' if 'amb_r_block' in v else '#3498db' for v in df['video_id']]
    
    ax.scatter(df['human_acc'], df['vlm_acc'], s=200, alpha=0.7, 
               c=colors, edgecolor='black', linewidth=2)
    
    # Add video labels
    for _, row in df.iterrows():
        label = row['video_id'].replace('_', '\n')
        ax.annotate(label, (row['human_acc'], row['vlm_acc']), 
                   fontsize=8, ha='center', va='bottom', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add diagonal line (y=x)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, linewidth=2, label='Equal Performance')
    
    # Add reference lines
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=50, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Human Accuracy (%)', fontweight='bold', fontsize=13)
    ax.set_ylabel('VLM Accuracy (%)', fontweight='bold', fontsize=13)
    ax.set_title('Human vs VLM Accuracy Per Video\n(Red = amb_r_block outlier)', 
                 fontweight='bold', fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add quadrant labels
    ax.text(75, 25, 'VLM\nWorse', ha='center', va='center', 
            fontsize=10, color='gray', alpha=0.5, style='italic')
    ax.text(25, 75, 'VLM\nBetter', ha='center', va='center', 
            fontsize=10, color='gray', alpha=0.5, style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_human_vs_vlm.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "scatter_human_vs_vlm.pdf", bbox_inches='tight')
    plt.close()
    print(f"✓ Scatter plot saved")

def main():
    """Generate all defense materials."""
    print("="*80)
    print("GENERATING THESIS DEFENSE MATERIALS")
    print("="*80)
    
    # Load data
    study_timepoints = load_user_study_timepoints()
    vlm_df = load_vlm_predictions(study_timepoints)
    human_df = load_human_decisions()
    
    # Generate materials
    create_confusion_matrices(human_df, vlm_df)
    statistical_tests(human_df, vlm_df)
    error_analysis(human_df, vlm_df)
    create_scatter_comparison(human_df, vlm_df)
    
    print("\n" + "="*80)
    print("✅ ALL DEFENSE MATERIALS GENERATED")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - confusion_matrices.png/pdf")
    print("  - statistical_tests.json")
    print("  - error_analysis.png/pdf")
    print("  - scatter_human_vs_vlm.png/pdf")
    print("\nUse these materials to support your defense!")

if __name__ == "__main__":
    main()
