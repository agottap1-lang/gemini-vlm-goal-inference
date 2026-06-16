#!/usr/bin/env python3
"""
Create VLM-Only Pipeline Diagram with Actual Robot Video Frames
Focused diagram showing only the VLM inference pipeline with real frame images
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
from PIL import Image

# Set academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# Paths
OUTPUT_DIR = Path(__file__).parent.parent / "analysis_results_2"
FRAMES_DIR = Path(__file__).parent.parent / "outputs" / "frames" / "amb_l_block"

# Color scheme - muted academic colors
COLOR_INPUT = '#E8EAF6'      # Light blue-gray
COLOR_PREPROCESS = '#FFF9C4'  # Light yellow
COLOR_MODEL = '#C8E6C9'       # Light green
COLOR_OUTPUT = '#FFCCBC'      # Light orange

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
        linewidth=2.0,
        linestyle=style,
        connectionstyle=f"arc3,rad={curvature}",
        zorder=1
    )
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.2, label,
                ha='center', va='bottom',
                fontsize=9, style='italic',
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
        linewidth=1.2,
        linestyle='--',
        alpha=0.6,
        zorder=0
    )
    ax.add_patch(box)
    
    # Add label at top
    ax.text(xy[0] + width/2, xy[1] + height + 0.2, label,
            ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            color=COLOR_BORDER,
            zorder=3)
    
    return box

def add_frame_image(ax, frame_path, xy, width, height, label=''):
    """Add an actual video frame image to the diagram."""
    try:
        img = Image.open(frame_path)
        
        # Create axes for the image
        img_ax = ax.inset_axes([xy[0], xy[1], width, height], 
                                transform=ax.transData)
        img_ax.imshow(img)
        img_ax.axis('off')
        
        # Add border
        rect = FancyBboxPatch(
            xy, width, height,
            boxstyle="round,pad=0.02",
            facecolor='none',
            edgecolor=COLOR_BORDER,
            linewidth=2,
            zorder=3,
            transform=ax.transData
        )
        ax.add_patch(rect)
        
        # Add label below
        if label:
            ax.text(xy[0] + width/2, xy[1] - 0.15, label,
                    ha='center', va='top',
                    fontsize=9, fontweight='bold',
                    color=COLOR_TEXT,
                    zorder=3)
        
        return img_ax
    except Exception as e:
        print(f"Warning: Could not load frame {frame_path}: {e}")
        # Draw placeholder box
        box = FancyBboxPatch(
            xy, width, height,
            facecolor='#EEEEEE',
            edgecolor=COLOR_BORDER,
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(box)
        ax.text(xy[0] + width/2, xy[1] + height/2, 'Frame',
                ha='center', va='center',
                fontsize=8, color=COLOR_BORDER)

def create_vlm_pipeline_with_frames():
    """Create VLM-only pipeline diagram with actual robot frames."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Title
    ax.text(8, 8.5, 'VLM Goal Inference Pipeline',
            ha='center', va='top',
            fontsize=16, fontweight='bold',
            color=COLOR_TEXT)
    
    # =====================================================================
    # STAGE 1: INPUT - Robot Motion Video with Actual Frames
    # =====================================================================
    create_module_group(ax, (0.3, 4.5), 3.8, 3.2, '① Input: Robot Motion Video')
    
    # Show actual frames from video
    timepoints = [0, 6, 8, 12]  # amb_l_block timepoints
    frame_width = 0.8
    frame_height = 0.8
    
    ax.text(2.2, 7.3, 'Video: amb_l_block (Ambiguous Trajectory)',
            ha='center', va='center',
            fontsize=9, style='italic', color=COLOR_BORDER)
    
    # Display frames
    for i, t in enumerate(timepoints):
        x_pos = 0.6 + (i * 0.95)
        y_pos = 5.8
        
        frame_path = FRAMES_DIR / f"t_{t:03d}.png"
        add_frame_image(ax, frame_path, (x_pos, y_pos), frame_width, frame_height, 
                       f't={t}s')
    
    # Video metadata
    ax.text(2.2, 5.2, '• Duration: 13 seconds\n• Goal: Move to left block (A)\n• Trajectory type: Ambiguous',
            ha='center', va='top',
            fontsize=8, color=COLOR_BORDER,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=COLOR_BORDER, linewidth=1, alpha=0.8))
    
    # =====================================================================
    # STAGE 2: PREPROCESSING - Frame Extraction & Timepoint Selection
    # =====================================================================
    create_module_group(ax, (4.5, 4.5), 3.8, 3.2, '② Preprocessing')
    
    # Frame extraction process
    create_box(ax, (4.8, 6.8), 3.2, 0.6,
               'Frame Extraction\n(1 FPS)', COLOR_PREPROCESS, fontweight='bold')
    
    # Show progression with arrows between frames
    ax.text(6.4, 6.3, '↓', ha='center', va='center',
            fontsize=20, fontweight='bold', color=COLOR_ARROW)
    
    # Timepoint selection
    create_box(ax, (4.8, 5.5), 3.2, 0.6,
               'Timepoint Selection\nt ∈ [0, 6, 8, 12]', COLOR_PREPROCESS, fontsize=9)
    
    # Cumulative frames indicator
    ax.text(6.4, 4.9, 'Cumulative sequence:\n{f₀, f₁, ..., fₜ}',
            ha='center', va='top',
            fontsize=8, style='italic', color=COLOR_BORDER,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7',
                     edgecolor=COLOR_BORDER, linewidth=1))
    
    # =====================================================================
    # STAGE 3: VLM INFERENCE - Vision-Language Model
    # =====================================================================
    create_module_group(ax, (8.7, 4.5), 4.0, 3.2, '③ VLM Inference')
    
    # Vision encoder
    create_box(ax, (9.0, 6.8), 3.4, 0.5,
               'Vision Encoder', COLOR_MODEL, fontsize=10)
    
    ax.text(10.7, 6.5, '(Extract visual features)', ha='center', va='top',
            fontsize=7, style='italic', color=COLOR_BORDER)
    
    # Multi-modal fusion
    create_box(ax, (9.0, 5.9), 3.4, 0.5,
               'Multi-Modal Fusion', COLOR_MODEL, fontweight='bold', fontsize=10)
    
    # Prompt
    create_box(ax, (9.0, 5.2), 3.4, 0.5,
               'Goal Inference Prompt', COLOR_MODEL, fontsize=9)
    
    ax.text(10.7, 4.9, '"What is the robot\'s goal?"', ha='center', va='top',
            fontsize=7, style='italic', color=COLOR_BORDER)
    
    # =====================================================================
    # STAGE 4: OUTPUT - Goal Prediction
    # =====================================================================
    create_module_group(ax, (13.1, 4.5), 2.6, 3.2, '④ Output')
    
    # Goal prediction
    create_box(ax, (13.4, 6.8), 2.0, 0.6,
               'Goal\nPrediction', COLOR_OUTPUT, fontweight='bold')
    
    # Predicted goal with confidence
    ax.text(14.4, 6.0, 'Predicted: A',
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color='#1B5E20',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9',
                     edgecolor='#1B5E20', linewidth=2))
    
    # Confidence
    ax.text(14.4, 5.4, 'Confidence: High',
            ha='center', va='center',
            fontsize=9, color=COLOR_BORDER)
    
    # Ground truth comparison
    ax.text(14.4, 4.8, '✓ Correct\n(Ground Truth: A)',
            ha='center', va='center',
            fontsize=8, fontweight='bold',
            color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                     edgecolor='#2E7D32', linewidth=1))
    
    # =====================================================================
    # ARROWS - Data Flow
    # =====================================================================
    
    # Input → Preprocessing
    create_arrow(ax, (4.1, 6.5), (4.8, 6.5), label='Raw video')
    
    # Preprocessing → VLM
    create_arrow(ax, (8.3, 6.5), (9.0, 6.5), label='Frame sequence')
    
    # VLM → Output
    create_arrow(ax, (12.4, 6.5), (13.4, 6.5), label='Inference')
    
    # Vertical flow in VLM
    ax.annotate('', xy=(10.7, 6.8), xytext=(10.7, 6.5),
                arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=1.5))
    ax.annotate('', xy=(10.7, 5.9), xytext=(10.7, 6.3),
                arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=1.5))
    ax.annotate('', xy=(10.7, 5.2), xytext=(10.7, 5.6),
                arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=1.5))
    
    # =====================================================================
    # BOTTOM: Goal Options & Performance
    # =====================================================================
    
    # Goal options
    y_bottom = 3.5
    ax.text(8, y_bottom, 'Goal Options:',
            ha='center', va='top',
            fontsize=11, fontweight='bold', color=COLOR_TEXT)
    
    goals_info = [
        ('A', 'Move to left block / Close bottom drawer', '#C8E6C9'),
        ('B', 'Move to right block / Close top drawer', '#FFCCBC'),
        ('C', 'Uncertain / Insufficient information', '#E0E0E0')
    ]
    
    for i, (goal, desc, color) in enumerate(goals_info):
        x_start = 2.5 + i * 4.2
        create_box(ax, (x_start, y_bottom - 0.7), 3.8, 0.5,
                   f'{goal}: {desc}', color, fontsize=8)
    
    # Performance metrics
    ax.text(8, y_bottom - 1.5, 'VLM Performance: 64.5% accuracy | 74.2% human agreement',
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color=COLOR_TEXT,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFDE7',
                     edgecolor=COLOR_BORDER, linewidth=2))
    
    # =====================================================================
    # ANNOTATIONS
    # =====================================================================
    
    # Key features
    features_text = (
        "Key Features:\n"
        "• Cumulative frame input (prefix frames f₀...fₜ)\n"
        "• Vision-Language Model: Google Gemini\n"
        "• Multi-modal reasoning (vision + text)\n"
        "• Real-time goal inference capability"
    )
    
    ax.text(0.5, 3.8, features_text,
            ha='left', va='top',
            fontsize=8, color=COLOR_BORDER,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                     edgecolor=COLOR_BORDER, linewidth=1, alpha=0.9))
    
    # Processing note
    ax.text(8, 8.0, 'Temporal Processing: Cumulative frames provide increasing context over time',
            ha='center', va='center',
            fontsize=9, style='italic',
            color='#0D47A1',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD',
                     edgecolor='#0D47A1', linewidth=1.5))
    
    plt.tight_layout()
    return fig

def create_simplified_vlm_flow():
    """Create a super clean, horizontal flow diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(7, 5.5, 'VLM Goal Inference: Data Flow',
            ha='center', va='top',
            fontsize=14, fontweight='bold',
            color=COLOR_TEXT)
    
    # Show 4 frames horizontally
    timepoints = [0, 6, 8, 12]
    frame_width = 1.2
    frame_height = 1.2
    
    # Frames section
    ax.text(2.4, 4.5, 'Input Frames', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=COLOR_BORDER)
    
    for i, t in enumerate(timepoints):
        x_pos = 0.5 + (i * 1.4)
        y_pos = 2.8
        
        frame_path = FRAMES_DIR / f"t_{t:03d}.png"
        add_frame_image(ax, frame_path, (x_pos, y_pos), frame_width, frame_height, 
                       f't={t}s')
    
    # Arrow
    create_arrow(ax, (6.2, 3.4), (7.5, 3.4), label='Vision encoder')
    
    # VLM box
    create_box(ax, (7.5, 2.5), 2.2, 1.8,
               'Vision-Language\nModel\n(Gemini)\n\nMulti-modal\nfusion', 
               COLOR_MODEL, fontweight='bold', fontsize=10)
    
    # Arrow
    create_arrow(ax, (9.7, 3.4), (10.8, 3.4), label='Classifier')
    
    # Output
    create_box(ax, (10.8, 2.8), 2.5, 1.2,
               'Goal: A\n(Left block)\n\nConfidence: High',
               COLOR_OUTPUT, fontweight='bold', fontsize=10)
    
    # Bottom annotations
    ax.text(7, 1.5, 
            'Cumulative frames: Each timepoint sees all previous frames (f₀, f₁, ..., fₜ)',
            ha='center', va='center',
            fontsize=9, style='italic', color=COLOR_ARROW,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=COLOR_ARROW, linewidth=1))
    
    ax.text(7, 0.8,
            'Result: 64.5% VLM accuracy | 74.2% agreement with humans',
            ha='center', va='center',
            fontsize=9, fontweight='bold', color=COLOR_TEXT,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFDE7',
                     edgecolor=COLOR_BORDER, linewidth=1.5))
    
    plt.tight_layout()
    return fig

def main():
    """Generate VLM-only pipeline diagrams with actual frames."""
    
    print("="*80)
    print("GENERATING VLM GOAL INFERENCE PIPELINE WITH ROBOT FRAMES")
    print("="*80)
    
    # Check if frames exist
    if not FRAMES_DIR.exists():
        print(f"\n⚠️  Warning: Frame directory not found: {FRAMES_DIR}")
        print("Creating diagrams with placeholder boxes...")
    else:
        frame_count = len(list(FRAMES_DIR.glob("*.png")))
        print(f"\n✓ Found {frame_count} frames in {FRAMES_DIR}")
    
    # Create main VLM pipeline with frames
    print("\n📊 Creating VLM pipeline diagram with robot frames...")
    fig1 = create_vlm_pipeline_with_frames()
    fig1.savefig(OUTPUT_DIR / "vlm_pipeline_with_frames.png", dpi=300, bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
    fig1.savefig(OUTPUT_DIR / "vlm_pipeline_with_frames.pdf", bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close(fig1)
    print("  ✓ Saved: vlm_pipeline_with_frames.png/pdf")
    
    # Create simplified flow diagram
    print("\n🔬 Creating simplified VLM flow diagram...")
    fig2 = create_simplified_vlm_flow()
    fig2.savefig(OUTPUT_DIR / "vlm_flow_simplified.png", dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig2.savefig(OUTPUT_DIR / "vlm_flow_simplified.pdf", bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close(fig2)
    print("  ✓ Saved: vlm_flow_simplified.png/pdf")
    
    print("\n" + "="*80)
    print("✅ VLM PIPELINE DIAGRAMS GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. vlm_pipeline_with_frames.png/pdf - Detailed 4-stage pipeline with robot frames")
    print("  2. vlm_flow_simplified.png/pdf - Clean horizontal flow diagram")
    print("\nFeatures:")
    print("  ✓ Actual robot video frames embedded (amb_l_block)")
    print("  ✓ Shows progression at t=0, 6, 8, 12 seconds")
    print("  ✓ VLM-only pipeline (no human baseline)")
    print("  ✓ Clean, focused design for technical presentations")
    print("  ✓ Publication-ready quality (300 DPI)")
    print("\nPerfect for:")
    print("  • Technical methodology sections")
    print("  • VLM architecture presentations")
    print("  • System overview slides")

if __name__ == "__main__":
    main()
