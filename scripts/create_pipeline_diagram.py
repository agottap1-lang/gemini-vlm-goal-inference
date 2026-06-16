#!/usr/bin/env python3
"""
Create ICRA-style Pipeline Flowchart for VLM Goal Inference System
Academic-quality diagram suitable for thesis/conference papers
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np
from pathlib import Path

# Set academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "analysis_results_2"

# Color scheme - muted academic colors
COLOR_INPUT = '#E8EAF6'      # Light blue-gray
COLOR_PREPROCESS = '#FFF9C4'  # Light yellow
COLOR_MODEL = '#C8E6C9'       # Light green
COLOR_OUTPUT = '#FFCCBC'      # Light orange
COLOR_HUMAN = '#F3E5F5'       # Light purple

COLOR_BORDER = '#424242'      # Dark gray
COLOR_ARROW = '#616161'       # Medium gray
COLOR_TEXT = '#212121'        # Almost black

def create_box(ax, xy, width, height, label, color, style='solid', fontsize=10, fontweight='normal'):
    """Create a box with text."""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor=COLOR_BORDER,
        linewidth=1.5 if style == 'solid' else 1.0,
        linestyle=style,
        zorder=2
    )
    ax.add_patch(box)
    
    # Add text
    center_x = xy[0] + width / 2
    center_y = xy[1] + height / 2
    ax.text(center_x, center_y, label,
            ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight,
            color=COLOR_TEXT,
            zorder=3)
    
    return box

def create_arrow(ax, start, end, style='-', label='', curvature=0):
    """Create a directional arrow."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->,head_width=0.4,head_length=0.6',
        color=COLOR_ARROW,
        linewidth=1.5,
        linestyle=style,
        connectionstyle=f"arc3,rad={curvature}",
        zorder=1
    )
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.15, label,
                ha='center', va='bottom',
                fontsize=8, style='italic',
                color=COLOR_ARROW,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='none', alpha=0.8),
                zorder=3)
    
    return arrow

def create_module_group(ax, xy, width, height, label):
    """Create a dashed box to group related components."""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.1",
        facecolor='none',
        edgecolor=COLOR_BORDER,
        linewidth=1.0,
        linestyle='--',
        alpha=0.6,
        zorder=0
    )
    ax.add_patch(box)
    
    # Add label at top
    ax.text(xy[0] + width/2, xy[1] + height + 0.15, label,
            ha='center', va='bottom',
            fontsize=11, fontweight='bold',
            color=COLOR_BORDER,
            zorder=3)
    
    return box

def create_pipeline_diagram():
    """Create the main pipeline flowchart."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'VLM-Based Goal Inference Pipeline',
            ha='center', va='top',
            fontsize=14, fontweight='bold',
            color=COLOR_TEXT)
    
    # =====================================================================
    # MODULE 1: INPUT STAGE
    # =====================================================================
    create_module_group(ax, (0.3, 6.5), 2.8, 2.2, 'Input')
    
    # Robot motion videos
    create_box(ax, (0.5, 7.8), 2.4, 0.6, 
               'Robot Motion\nVideos', COLOR_INPUT, fontweight='bold')
    
    # Video metadata
    create_box(ax, (0.5, 7.0), 2.4, 0.5,
               '8 videos\n(4 amb., 4 leg.)', COLOR_INPUT, fontsize=9)
    
    # =====================================================================
    # MODULE 2: PREPROCESSING
    # =====================================================================
    create_module_group(ax, (3.7, 6.5), 2.8, 2.2, 'Preprocessing')
    
    # Frame extraction
    create_box(ax, (3.9, 7.8), 2.4, 0.6,
               'Frame\nExtraction', COLOR_PREPROCESS, fontweight='bold')
    
    # Timepoint selection
    create_box(ax, (3.9, 7.0), 2.4, 0.5,
               'Timepoint\nSelection', COLOR_PREPROCESS, fontsize=9)
    
    # Details
    ax.text(5.1, 6.65, 't = [0, ..., T]',
            ha='center', va='top',
            fontsize=8, style='italic',
            color=COLOR_BORDER)
    
    # =====================================================================
    # MODULE 3: VLM INFERENCE
    # =====================================================================
    create_module_group(ax, (7.1, 6.0), 3.2, 2.7, 'VLM Module')
    
    # Cumulative frames
    create_box(ax, (7.3, 7.8), 2.8, 0.6,
               'Cumulative\nFrame Sequence', COLOR_MODEL, fontweight='bold')
    
    # VLM (Gemini)
    create_box(ax, (7.3, 6.9), 2.8, 0.7,
               'Vision-Language\nModel (Gemini)', COLOR_MODEL, 
               fontweight='bold', fontsize=10)
    
    # Prompt
    create_box(ax, (7.3, 6.2), 2.8, 0.5,
               'Goal Inference\nPrompt', COLOR_MODEL, fontsize=9)
    
    # =====================================================================
    # MODULE 4: OUTPUT & DECISION
    # =====================================================================
    create_module_group(ax, (10.9, 6.5), 2.8, 2.2, 'Output')
    
    # Goal prediction
    create_box(ax, (11.1, 7.8), 2.4, 0.6,
               'Goal Prediction\n{A, B, C}', COLOR_OUTPUT, fontweight='bold')
    
    # Confidence score
    create_box(ax, (11.1, 7.0), 2.4, 0.5,
               'Confidence\nScore', COLOR_OUTPUT, fontsize=9)
    
    # =====================================================================
    # MODULE 5: HUMAN BASELINE (PARALLEL PATH)
    # =====================================================================
    create_module_group(ax, (3.7, 3.5), 6.6, 2.2, 'Human Baseline')
    
    # Human participants
    create_box(ax, (3.9, 4.8), 2.4, 0.6,
               'Human\nParticipants', COLOR_HUMAN, fontweight='bold')
    
    # Study interface
    create_box(ax, (6.8, 4.8), 2.4, 0.6,
               'User Study\nInterface', COLOR_HUMAN, fontweight='bold')
    
    # Human decisions
    create_box(ax, (9.7, 4.8), 2.4, 0.6,
               'Human Goal\nPredictions', COLOR_HUMAN, fontweight='bold')
    
    # Details
    ax.text(5.1, 4.0, 'N = 8 participants',
            ha='center', va='top',
            fontsize=8, style='italic',
            color=COLOR_BORDER)
    
    ax.text(8.0, 4.0, 'Same timepoints',
            ha='center', va='top',
            fontsize=8, style='italic',
            color=COLOR_BORDER)
    
    # =====================================================================
    # MODULE 6: EVALUATION
    # =====================================================================
    create_module_group(ax, (3.7, 0.8), 6.6, 2.0, 'Evaluation')
    
    # Accuracy
    create_box(ax, (4.2, 1.8), 1.8, 0.5,
               'Accuracy vs\nGround Truth', COLOR_OUTPUT, fontsize=9)
    
    # IoU
    create_box(ax, (6.5, 1.8), 1.8, 0.5,
               'Human-VLM\nAgreement (IoU)', COLOR_OUTPUT, fontsize=9)
    
    # Timing (optional)
    create_box(ax, (8.8, 1.8), 1.8, 0.5,
               'Inference\nTiming', COLOR_OUTPUT, fontsize=9)
    
    # Results
    ax.text(7.0, 1.0, 'Human: 66.7%  |  VLM: 64.5%  |  IoU: 74.2%',
            ha='center', va='top',
            fontsize=9, fontweight='bold',
            color=COLOR_TEXT,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFDE7',
                     edgecolor=COLOR_BORDER, linewidth=1))
    
    # =====================================================================
    # ARROWS - MAIN PIPELINE FLOW
    # =====================================================================
    
    # Input → Preprocessing
    create_arrow(ax, (2.9, 8.1), (3.9, 8.1))
    
    # Preprocessing → VLM
    create_arrow(ax, (6.3, 8.1), (7.3, 8.1))
    
    # VLM → Output
    create_arrow(ax, (10.1, 8.1), (11.1, 8.1))
    
    # Input → Human (parallel path)
    create_arrow(ax, (1.7, 6.5), (1.7, 5.8), style='--')
    create_arrow(ax, (1.7, 5.8), (3.9, 5.1), style='--', curvature=0.2)
    
    # Preprocessing → Human interface
    create_arrow(ax, (5.1, 6.5), (5.1, 5.8), style='--')
    create_arrow(ax, (5.1, 5.8), (6.8, 5.4), style='--', curvature=0.2)
    
    # Human participants → interface
    create_arrow(ax, (6.3, 5.1), (6.8, 5.1), style='--')
    
    # Interface → Human predictions
    create_arrow(ax, (9.2, 5.1), (9.7, 5.1), style='--')
    
    # VLM output → Evaluation
    create_arrow(ax, (12.3, 6.5), (12.3, 3.5), style='-')
    create_arrow(ax, (12.3, 3.5), (9.1, 2.5), style='-', curvature=-0.3)
    
    # Human output → Evaluation
    create_arrow(ax, (11.0, 4.5), (11.0, 3.5), style='--')
    create_arrow(ax, (11.0, 3.5), (9.1, 2.5), style='--', curvature=0.3)
    
    # Ground truth → Evaluation
    ax.text(2.5, 2.5, 'Ground\nTruth',
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color=COLOR_TEXT,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0E0E0',
                     edgecolor=COLOR_BORDER, linewidth=1))
    create_arrow(ax, (3.2, 2.5), (4.2, 2.2), style='-')
    
    # =====================================================================
    # ANNOTATIONS
    # =====================================================================
    
    # Key operations
    ax.annotate('', xy=(5.1, 8.5), xytext=(5.1, 9.0),
                arrowprops=dict(arrowstyle='-', color='none'))
    ax.text(6.5, 8.65, 'Prefix frames: f₀, f₁, ..., fₜ',
            fontsize=8, style='italic', color=COLOR_ARROW)
    
    ax.text(8.7, 7.5, 'Multi-modal\nprocessing',
            fontsize=7, style='italic', color=COLOR_ARROW,
            ha='center')
    
    # Fair comparison note
    ax.text(7.0, 5.7, '⚠ Fair Comparison: Same timepoints for VLM & Human',
            ha='center', va='center',
            fontsize=8, fontweight='bold',
            color='#D32F2F',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE',
                     edgecolor='#D32F2F', linewidth=1.5, linestyle='--'))
    
    # Legend
    legend_y = 0.3
    ax.text(0.5, legend_y, 'Legend:',
            fontsize=9, fontweight='bold', color=COLOR_TEXT)
    
    # Solid arrow
    create_arrow(ax, (1.0, legend_y - 0.15), (1.8, legend_y - 0.15), style='-')
    ax.text(2.1, legend_y - 0.15, 'VLM pipeline',
            fontsize=8, va='center', color=COLOR_TEXT)
    
    # Dashed arrow
    create_arrow(ax, (1.0, legend_y - 0.35), (1.8, legend_y - 0.35), style='--')
    ax.text(2.1, legend_y - 0.35, 'Human baseline',
            fontsize=8, va='center', color=COLOR_TEXT)
    
    plt.tight_layout()
    return fig

def create_detailed_vlm_architecture():
    """Create a detailed view of VLM architecture."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'VLM Architecture: Goal Inference from Video Frames',
            ha='center', va='top',
            fontsize=13, fontweight='bold',
            color=COLOR_TEXT)
    
    # =====================================================================
    # Input Stage
    # =====================================================================
    
    # Frame sequence
    for i in range(4):
        x_pos = 0.5 + i * 1.0
        create_box(ax, (x_pos, 6.0), 0.8, 0.8,
                   f'f_{i}', COLOR_INPUT, fontsize=9)
    
    ax.text(2.5, 6.4, '...', ha='center', va='center',
            fontsize=16, fontweight='bold')
    
    create_box(ax, (3.1, 6.0), 0.8, 0.8,
               f'f_t', COLOR_INPUT, fontsize=9)
    
    ax.text(2.3, 5.5, 'Cumulative Frame Sequence',
            ha='center', va='top',
            fontsize=10, style='italic', color=COLOR_BORDER)
    
    # =====================================================================
    # Vision Encoder
    # =====================================================================
    
    create_module_group(ax, (0.3, 3.8), 4.2, 1.5, 'Vision Encoder')
    
    create_box(ax, (0.6, 4.3), 1.8, 0.6,
               'Image\nEmbedding', COLOR_PREPROCESS, fontsize=9)
    
    create_arrow(ax, (2.3, 6.0), (1.5, 4.9))
    
    create_box(ax, (2.7, 4.3), 1.5, 0.6,
               'Visual\nFeatures', COLOR_PREPROCESS, fontsize=9)
    
    create_arrow(ax, (2.4, 4.3), (2.7, 4.6))
    
    # =====================================================================
    # Language Model / Reasoning
    # =====================================================================
    
    create_module_group(ax, (5.0, 3.0), 4.5, 2.3, 'Language Model')
    
    # Prompt input
    create_box(ax, (5.3, 4.6), 1.8, 0.5,
               'Task Prompt', COLOR_MODEL, fontsize=9)
    
    ax.text(6.2, 4.3, '"What is the robot\'s goal?"',
            ha='center', va='top',
            fontsize=7, style='italic', color=COLOR_BORDER)
    
    # Multi-modal fusion
    create_box(ax, (5.3, 3.6), 3.9, 0.6,
               'Multi-Modal Fusion (Vision + Language)', 
               COLOR_MODEL, fontsize=9, fontweight='bold')
    
    # Transformer layers
    create_box(ax, (5.3, 3.0), 3.9, 0.4,
               'Self-Attention Layers', COLOR_MODEL, fontsize=8)
    
    # Arrows
    create_arrow(ax, (4.2, 4.6), (5.3, 4.8))
    create_arrow(ax, (6.2, 4.6), (6.2, 4.2))
    create_arrow(ax, (7.2, 4.2), (7.2, 3.6))
    
    # =====================================================================
    # Output Stage
    # =====================================================================
    
    create_module_group(ax, (10.0, 3.5), 1.7, 1.8, 'Output')
    
    create_box(ax, (10.2, 4.6), 1.3, 0.5,
               'Classifier', COLOR_OUTPUT, fontsize=9)
    
    create_box(ax, (10.2, 3.8), 1.3, 0.5,
               'Goal:\nA/B/C', COLOR_OUTPUT, fontsize=9, fontweight='bold')
    
    create_arrow(ax, (9.2, 3.9), (10.2, 4.8))
    create_arrow(ax, (10.85, 4.6), (10.85, 4.3))
    
    # =====================================================================
    # Goal Options
    # =====================================================================
    
    ax.text(6.0, 1.8, 'Goal Options:',
            ha='left', va='top',
            fontsize=10, fontweight='bold', color=COLOR_TEXT)
    
    goals = [
        ('A', 'Move to left block / Close bottom drawer'),
        ('B', 'Move to right block / Close top drawer'),
        ('C', 'Uncertain / Insufficient information')
    ]
    
    for i, (goal, desc) in enumerate(goals):
        y_pos = 1.4 - i * 0.25
        ax.text(6.0, y_pos, f'  {goal}:', ha='left', va='center',
                fontsize=9, fontweight='bold', color=COLOR_TEXT)
        ax.text(6.5, y_pos, desc, ha='left', va='center',
                fontsize=8, color=COLOR_BORDER)
    
    # =====================================================================
    # Additional annotations
    # =====================================================================
    
    ax.text(6.0, 2.3, 'Trajectory Types: Legible vs Ambiguous',
            ha='center', va='center',
            fontsize=9, style='italic',
            color=COLOR_ARROW,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                     edgecolor=COLOR_BORDER, linewidth=1))
    
    plt.tight_layout()
    return fig

def main():
    """Generate all pipeline diagrams."""
    
    print("="*80)
    print("GENERATING ICRA-STYLE PIPELINE DIAGRAMS")
    print("="*80)
    
    # Create main pipeline diagram
    print("\n📊 Creating main pipeline flowchart...")
    fig1 = create_pipeline_diagram()
    fig1.savefig(OUTPUT_DIR / "pipeline_diagram.png", dpi=300, bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
    fig1.savefig(OUTPUT_DIR / "pipeline_diagram.pdf", bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close(fig1)
    print("  ✓ Saved: pipeline_diagram.png/pdf")
    
    # Create detailed VLM architecture
    print("\n🔬 Creating detailed VLM architecture...")
    fig2 = create_detailed_vlm_architecture()
    fig2.savefig(OUTPUT_DIR / "vlm_architecture_detailed.png", dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig2.savefig(OUTPUT_DIR / "vlm_architecture_detailed.pdf", bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close(fig2)
    print("  ✓ Saved: vlm_architecture_detailed.png/pdf")
    
    print("\n" + "="*80)
    print("✅ DIAGRAMS GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. pipeline_diagram.png/pdf - Full system pipeline")
    print("  2. vlm_architecture_detailed.png/pdf - Detailed VLM architecture")
    print("\nThese diagrams are ready for inclusion in:")
    print("  • ICRA papers")
    print("  • Thesis chapters")
    print("  • Presentations")
    print("  • ArXiv submissions")
    print("\nStyle features:")
    print("  ✓ Clean, minimal design")
    print("  ✓ Academic color scheme")
    print("  ✓ Proper typography (serif font)")
    print("  ✓ Modular component grouping")
    print("  ✓ Clear data flow arrows")
    print("  ✓ Publication-ready quality (300 DPI)")

if __name__ == "__main__":
    main()
