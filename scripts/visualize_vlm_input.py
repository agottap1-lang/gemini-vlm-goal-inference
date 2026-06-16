#!/usr/bin/env python3
"""
Visualize what the VLM sees at a specific timepoint in different evaluation modes.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Setup
frames_dir = Path('outputs/frames/amb_l_block')
output_dir = Path('analysis_results_final')
output_dir.mkdir(exist_ok=True)

# Create figure showing what model sees at t=5
fig = plt.figure(figsize=(16, 10))
fig.suptitle('What the VLM Sees at t=5 seconds: amb_l_block video', 
             fontsize=16, fontweight='bold', y=0.98)

# MODE 1: single_frame (top row)
ax1 = plt.subplot(2, 6, 4)
img = mpimg.imread(frames_dir / 't_005.png')
ax1.imshow(img)
ax1.set_title('single_frame mode\n\nAt t=5s: ONLY frame 5', 
              fontsize=12, fontweight='bold', color='#D32F2F')
ax1.axis('off')

# Add text explanation
fig.text(0.5, 0.52, 'Mode 1: single_frame - Model sees ONLY 1 frame (memoryless)', 
         ha='center', fontsize=13, fontweight='bold', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=2))

# MODE 2: prefix_frames (bottom row)
fig.text(0.5, 0.47, 'Mode 2: prefix_frames - Model sees ALL 6 frames from t=0 to t=5 (cumulative motion)', 
         ha='center', fontsize=13, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2))

for i, t in enumerate([0, 1, 2, 3, 4, 5]):
    ax = plt.subplot(2, 6, 7 + i)
    img = mpimg.imread(frames_dir / f't_{t:03d}.png')
    ax.imshow(img)
    ax.set_title(f'Frame {t}\n(t={t}s)', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    if i == 5:
        # Highlight the final frame
        for spine in ax.spines.values():
            spine.set_edgecolor('#4CAF50')
            spine.set_linewidth(4)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_dir / 'vlm_input_comparison_t5.png', dpi=200, bbox_inches='tight')
print('✓ Saved visualization to analysis_results_final/vlm_input_comparison_t5.png')
print()
print('=' * 80)
print('WHAT THE VLM SEES AT t=5 seconds (amb_l_block video):')
print('=' * 80)
print()
print('📸 single_frame mode:')
print('   - Inputs: 1 image (frame at t=5)')
print('   - Frames shown: 1 frame')
print('   - Context: NONE (memoryless snapshot)')
print('   - Model instruction: "Use ONLY this frame"')
print()
print('🎬 prefix_frames mode:')
print('   - Inputs: 6 images (frames 0,1,2,3,4,5 in order)')
print('   - Frames shown: 6 frames')
print('   - Context: FULL motion history from start')
print('   - Model instruction: "You have observed motion from t=0 to t=5s"')
print()
print('=' * 80)
print('VLM PREDICTION AT t=5 (prefix_frames mode):')
print('=' * 80)
print('  pA = 0.75, pB = 0.25')
print('  Choice: A (pick left block)')
print('  Confidence: 75%')
print('  Cue: "robot arm moving towards the left block"')
print('  Legible: YES (legible_now)')
print()
print('✓ Ground truth: A (correct prediction!)')
