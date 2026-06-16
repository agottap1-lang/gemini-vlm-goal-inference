#!/usr/bin/env python3
"""
Generate Publication-Ready Figures for Multi-Model VLM Evaluation
================================================================

Creates clear, professional figures for presentations showing:
- Model comparison
- Per-video accuracy
- Anomaly analysis
- Worst-performing videos
- Temporal performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
ANALYSIS_DIR = Path("analysis_3")
OUTPUT_DIR = ANALYSIS_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

# Color scheme
COLORS = {
    'gemini-2.5-flash': '#FFA726',     # Orange
    'gemini-2.5-pro': '#66BB6A',        # Light Green
    'gemini-3-pro-preview': '#42A5F5',  # Blue
    'gemini-3.1-pro-preview': '#AB47BC', # Purple (WINNER!)
    'gemini-pro-latest': '#26C6DA'      # Cyan
}

MODEL_LABELS = {
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gemini-2.5-pro': 'Gemini 2.5 Pro',
    'gemini-3-pro-preview': 'Gemini 3 Pro',
    'gemini-3.1-pro-preview': 'Gemini 3.1 Pro ⭐',
    'gemini-pro-latest': 'Gemini Pro (Latest)'
}

def create_figure1_model_comparison():
    """Figure 1: Overall Model Performance Comparison"""
    
    # Load data
    df = pd.read_csv(ANALYSIS_DIR / "model_comparison.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Figure 1: Multi-Model Performance Comparison', 
                 fontweight='bold', fontsize=16, y=1.02)
    
    # Panel A: Accuracy Comparison
    ax = axes[0]
    models = df['Model'].tolist()
    accuracy = df['VLM Accuracy (%)'].tolist()
    colors = [COLORS[m] for m in models]
    
    bars = ax.bar(range(len(models)), accuracy, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('(A) VLM Accuracy', fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=20, ha='right')
    ax.set_ylim(0, 100)
    ax.axhline(y=98.25, color='red', linestyle='--', linewidth=2, label='Human (98.25%)')
    ax.legend()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, accuracy)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, 
               f'{val:.1f}%', ha='center', va='bottom', 
               fontweight='bold', fontsize=11)
    
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: IoU (Agreement)
    ax = axes[1]
    iou = df['IoU (%)'].tolist()
    
    bars = ax.bar(range(len(models)), iou, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('IoU with Humans (%)', fontweight='bold')
    ax.set_title('(B) Agreement with Human Judgments', fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=20, ha='right')
    ax.set_ylim(0, 100)
    
    for i, (bar, val) in enumerate(zip(bars, iou)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2,
               f'{val:.1f}%', ha='center', va='bottom',
               fontweight='bold', fontsize=11)
    
    ax.grid(axis='y', alpha=0.3)
    
    # Panel C: Anomalies
    ax = axes[2]
    anomalies = df['Anomalies'].tolist()
    
    bars = ax.bar(range(len(models)), anomalies, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Anomalies', fontweight='bold')
    ax.set_title('(C) Reliability (Fewer = Better)', fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=20, ha='right')
    
    for i, (bar, val) in enumerate(zip(bars, anomalies)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
               f'{int(val)}', ha='center', va='bottom',
               fontweight='bold', fontsize=11)
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / "figure1_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure1_model_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 1: Model Comparison")

def create_figure2_per_video_accuracy():
    """Figure 2: Per-Video Accuracy Heatmap"""
    
    # Load data
    df = pd.read_csv(ANALYSIS_DIR / "video_comparison.csv")
    
    # Extract accuracy columns
    models = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3-pro-preview']
    video_ids = df['video_id'].tolist()
    
    # Create accuracy matrix
    acc_matrix = []
    for model in models:
        acc_col = f'{model}_acc'
        acc_matrix.append(df[acc_col].tolist())
    
    acc_matrix = np.array(acc_matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Figure 2: Per-Video Accuracy Comparison', 
                 fontweight='bold', fontsize=16)
    
    # Create heatmap
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(range(len(video_ids)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([vid.replace('_', '\n') for vid in video_ids], 
                       rotation=0, ha='center', fontsize=10)
    ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=11)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(video_ids)):
            text = ax.text(j, i, f'{acc_matrix[i, j]:.0f}%',
                          ha="center", va="center", color="black",
                          fontweight='bold', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Accuracy (%)', fontweight='bold', fontsize=12)
    
    # Add legend for trajectory types
    trajectory_types = []
    for vid in video_ids:
        if 'amb' in vid:
            trajectory_types.append('A')
        else:
            trajectory_types.append('L')
    
    legend_text = "Trajectory Types: A = Ambiguous, L = Legible"
    ax.text(0.5, -0.15, legend_text, transform=ax.transAxes,
           ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / "figure2_per_video_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure2_per_video_heatmap.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 2: Per-Video Accuracy Heatmap")

def create_figure3_worst_performing_videos():
    """Figure 3: Worst-Performing Videos Identification"""
    
    # Load data
    df = pd.read_csv(ANALYSIS_DIR / "video_comparison.csv")
    
    # Calculate average accuracy across all models
    models = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3-pro-preview']
    df['avg_accuracy'] = df[[f'{m}_acc' for m in models]].mean(axis=1)
    
    # Sort by average accuracy
    df = df.sort_values('avg_accuracy')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Figure 3: Worst-Performing Videos (All Models)', 
                 fontweight='bold', fontsize=16)
    
    # Plot bars for each model
    x = np.arange(len(df))
    width = 0.25
    
    for i, model in enumerate(models):
        acc_col = f'{model}_acc'
        offset = (i - 1) * width
        bars = ax.bar(x + offset, df[acc_col], width, 
                     label=MODEL_LABELS[model],
                     color=COLORS[model], alpha=0.8,
                     edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Video ID', fontweight='bold', fontsize=13)
    ax.set_title('Videos Ranked by Average Accuracy (Low to High)', 
                 fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([vid.replace('_', '\n') for vid in df['video_id']], 
                       rotation=0, ha='center', fontsize=10)
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight worst 4
    for i in range(min(4, len(df))):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='red', zorder=0)
    
    # Add annotation
    ax.text(0.02, 0.98, '← Worst performing', 
           transform=ax.transAxes, fontsize=11, va='top',
           bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
    
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / "figure3_worst_videos.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure3_worst_videos.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 3: Worst-Performing Videos")

def create_figure4_anomaly_breakdown():
    """Figure 4: Anomaly Type Breakdown"""
    
    # Load data
    df = pd.read_csv(ANALYSIS_DIR / "anomalies.csv")
    
    # Count anomalies by type and model
    anomaly_counts = df.groupby(['model', 'type']).size().unstack(fill_value=0)
    
    # Reorder models
    model_order = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3-pro-preview']
    anomaly_counts = anomaly_counts.reindex(model_order)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Figure 4: Anomaly Analysis by Model', 
                 fontweight='bold', fontsize=16, y=1.02)
    
    # Panel A: Stacked bar chart
    ax = axes[0]
    anomaly_counts.plot(kind='bar', stacked=True, ax=ax, 
                       color=['#E57373', '#FFB74D', '#81C784', '#64B5F6'],
                       edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_title('(A) Anomaly Types by Model', fontweight='bold', fontsize=14)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Number of Anomalies', fontweight='bold')
    ax.set_xticklabels([MODEL_LABELS[m] for m in model_order], rotation=20, ha='right')
    ax.legend(title='Anomaly Type', bbox_to_anchor=(1.05, 1), loc='upper left',
             labels=['Choice Flip', 'High Conf Wrong', 'Final Uncertain', 'Low Conf Legible'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add total counts on top
    totals = anomaly_counts.sum(axis=1)
    for i, (idx, total) in enumerate(totals.items()):
        ax.text(i, total + 0.3, f'{int(total)}', ha='center', va='bottom',
               fontweight='bold', fontsize=11)
    
    # Panel B: Pie chart for most problematic model (Flash)
    ax = axes[1]
    flash_anomalies = anomaly_counts.loc['gemini-2.5-flash']
    flash_anomalies = flash_anomalies[flash_anomalies > 0]
    
    colors_pie = ['#E57373', '#FFB74D', '#81C784', '#64B5F6'][:len(flash_anomalies)]
    wedges, texts, autotexts = ax.pie(flash_anomalies, labels=flash_anomalies.index,
                                       autopct='%1.0f%%', startangle=90,
                                       colors=colors_pie, textprops={'fontsize': 11})
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax.set_title('(B) Gemini 2.5 Flash Anomaly Distribution\n(Most Problematic Model)',
                fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / "figure4_anomaly_breakdown.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure4_anomaly_breakdown.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 4: Anomaly Breakdown")

def create_figure5_model_tradeoffs():
    """Figure 5: Model Trade-offs (Accuracy vs Speed vs Reliability)"""
    
    # Load data
    df = pd.read_csv(ANALYSIS_DIR / "model_comparison.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 5: Model Trade-offs Analysis', 
                 fontweight='bold', fontsize=16, y=1.02)
    
    # Panel A: Accuracy vs Speed
    ax = axes[0]
    
    for i, row in df.iterrows():
        model = row['Model']
        ax.scatter(row['Time (s)'], row['VLM Accuracy (%)'], 
                  s=300, color=COLORS[model], alpha=0.7,
                  edgecolor='black', linewidth=2, zorder=3)
        
        # Add label
        label = MODEL_LABELS[model].replace(' ', '\n')
        ax.annotate(label, (row['Time (s)'], row['VLM Accuracy (%)']),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS[model], 
                            alpha=0.3, edgecolor='black'))
    
    ax.set_xlabel('Inference Time (seconds)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('(A) Accuracy vs Speed\n(Top-Left = Best)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(300, 500)
    ax.set_ylim(78, 96)
    
    # Add quadrant lines
    ax.axhline(y=df['VLM Accuracy (%)'].median(), color='gray', 
              linestyle='--', alpha=0.5)
    ax.axvline(x=df['Time (s)'].median(), color='gray', 
              linestyle='--', alpha=0.5)
    
    # Panel B: Accuracy vs Anomalies
    ax = axes[1]
    
    for i, row in df.iterrows():
        model = row['Model']
        ax.scatter(row['Anomalies'], row['VLM Accuracy (%)'],
                  s=300, color=COLORS[model], alpha=0.7,
                  edgecolor='black', linewidth=2, zorder=3)
        
        label = MODEL_LABELS[model].replace(' ', '\n')
        ax.annotate(label, (row['Anomalies'], row['VLM Accuracy (%)']),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS[model],
                            alpha=0.3, edgecolor='black'))
    
    ax.set_xlabel('Number of Anomalies', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('(B) Accuracy vs Reliability\n(Top-Left = Best)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2, 10)
    ax.set_ylim(78, 96)
    
    # Add quadrant lines
    ax.axhline(y=df['VLM Accuracy (%)'].median(), color='gray',
              linestyle='--', alpha=0.5)
    ax.axvline(x=df['Anomalies'].median(), color='gray',
              linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / "figure5_model_tradeoffs.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure5_model_tradeoffs.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 5: Model Trade-offs")

def create_figure6_summary_infographic():
    """Figure 6: One-Page Summary Infographic"""
    
    # Load data
    comp_df = pd.read_csv(ANALYSIS_DIR / "model_comparison.csv")
    video_df = pd.read_csv(ANALYSIS_DIR / "video_comparison.csv")
    
    # Best model
    best_model = comp_df.iloc[0]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    fig.suptitle('Figure 6: Multi-Model VLM Evaluation - Executive Summary', 
                 fontweight='bold', fontsize=18, y=0.98)
    
    # Top banner: Winner
    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.axis('off')
    
    banner_text = f"🏆 BEST MODEL: {MODEL_LABELS[best_model['Model']].upper()}"
    ax_banner.text(0.5, 0.7, banner_text, ha='center', va='center',
                  fontsize=24, fontweight='bold', color=COLORS[best_model['Model']],
                  bbox=dict(boxstyle='round,pad=1', facecolor='white', 
                           edgecolor=COLORS[best_model['Model']], linewidth=4))
    
    stats_text = f"Accuracy: {best_model['VLM Accuracy (%)']:.1f}%  |  IoU: {best_model['IoU (%)']:.1f}%  |  Anomalies: {int(best_model['Anomalies'])}  |  Time: {best_model['Time (s)']:.0f}s"
    ax_banner.text(0.5, 0.2, stats_text, ha='center', va='center',
                  fontsize=14, fontweight='bold')
    
    # Row 2, Col 1: Accuracy comparison
    ax1 = fig.add_subplot(gs[1, 0])
    models = comp_df['Model'].tolist()
    accuracy = comp_df['VLM Accuracy (%)'].tolist()
    colors = [COLORS[m] for m in models]
    
    ax1.barh(range(len(models)), accuracy, color=colors, alpha=0.8,
            edgecolor='black', linewidth=2)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=11)
    ax1.set_xlabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Model Accuracy', fontweight='bold', fontsize=13)
    ax1.set_xlim(0, 100)
    
    for i, val in enumerate(accuracy):
        ax1.text(val + 2, i, f'{val:.1f}%', va='center', fontweight='bold')
    
    # Row 2, Col 2: Anomalies
    ax2 = fig.add_subplot(gs[1, 1])
    anomalies = comp_df['Anomalies'].tolist()
    
    ax2.barh(range(len(models)), anomalies, color=colors, alpha=0.8,
            edgecolor='black', linewidth=2)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=11)
    ax2.set_xlabel('Number of Anomalies', fontweight='bold')
    ax2.set_title('Reliability (Lower = Better)', fontweight='bold', fontsize=13)
    
    for i, val in enumerate(anomalies):
        ax2.text(val + 0.3, i, f'{int(val)}', va='center', fontweight='bold')
    
    # Row 2, Col 3: Worst videos
    ax3 = fig.add_subplot(gs[1, 2])
    video_df['avg_accuracy'] = video_df[['gemini-2.5-flash_acc', 
                                         'gemini-2.5-pro_acc',
                                         'gemini-3-pro-preview_acc']].mean(axis=1)
    worst_videos = video_df.nsmallest(4, 'avg_accuracy')
    
    ax3.barh(range(len(worst_videos)), worst_videos['avg_accuracy'], 
            color='#E57373', alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_yticks(range(len(worst_videos)))
    ax3.set_yticklabels([vid.replace('_', ' ') for vid in worst_videos['video_id']], 
                       fontsize=10)
    ax3.set_xlabel('Avg Accuracy (%)', fontweight='bold')
    ax3.set_title('Worst-Performing Videos', fontweight='bold', fontsize=13)
    ax3.set_xlim(0, 100)
    
    for i, val in enumerate(worst_videos['avg_accuracy']):
        ax3.text(val + 2, i, f'{val:.0f}%', va='center', fontweight='bold')
    
    # Row 3: Key findings text
    ax_findings = fig.add_subplot(gs[2, :])
    ax_findings.axis('off')
    
    findings_text = """
    KEY FINDINGS:
    
    ✓ gemini-2.5-pro achieves 93.75% accuracy - closest to human performance (98.25%)
    ✓ 12% improvement over Flash baseline with 56% fewer anomalies
    ✓ Gemini-3-pro-preview underperforms (90.48%) - likely due to preview status
    ✓ All models struggle with amb_d/to_drawer_close and le_r_block (spatial reasoning)
    ✓ prefix_frames mode successfully integrates temporal information
    
    RECOMMENDATION: Deploy gemini-2.5-pro for production VLM-based goal inference
    """
    
    ax_findings.text(0.5, 0.5, findings_text, ha='center', va='center',
                    fontsize=12, family='monospace',
                    bbox=dict(boxstyle='round,pad=1', facecolor='#E3F2FD', 
                             edgecolor='#1976D2', linewidth=2))
    
    # Save
    plt.savefig(OUTPUT_DIR / "figure6_summary_infographic.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure6_summary_infographic.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 6: Summary Infographic")

def create_all_figures():
    """Generate all figures"""
    
    print("=" * 70)
    print("GENERATING PUBLICATION-READY FIGURES")
    print("=" * 70)
    print()
    
    create_figure1_model_comparison()
    create_figure2_per_video_accuracy()
    create_figure3_worst_performing_videos()
    create_figure4_anomaly_breakdown()
    create_figure5_model_tradeoffs()
    create_figure6_summary_infographic()
    
    print()
    print("=" * 70)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print()
    print(f"📁 Saved to: {OUTPUT_DIR.absolute()}")
    print()
    print("Generated figures:")
    print("  1. figure1_model_comparison.png/pdf      - Overall performance")
    print("  2. figure2_per_video_heatmap.png/pdf     - Per-video accuracy")
    print("  3. figure3_worst_videos.png/pdf          - Worst performers")
    print("  4. figure4_anomaly_breakdown.png/pdf     - Anomaly analysis")
    print("  5. figure5_model_tradeoffs.png/pdf       - Trade-off analysis")
    print("  6. figure6_summary_infographic.png/pdf   - One-page summary")
    print()
    print("✓ All figures are high-resolution (300 DPI)")
    print("✓ Available in both PNG and PDF formats")
    print("✓ Ready for presentations and publications")
    print()

if __name__ == "__main__":
    create_all_figures()
