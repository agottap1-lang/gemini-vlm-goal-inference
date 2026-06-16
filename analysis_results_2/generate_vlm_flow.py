# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(20, 8))
ax.set_xlim(0, 20)
ax.set_ylim(0.5, 7.5)
ax.axis('off')

pipe_y = 4.5       # vertical centre of pipeline
box_w  = 1.05
box_h  = 1.9
ox, oy = 0.16, 0.16   # diagonal stack offset (back boxes up-right)

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(10, 7.3, 'VLM Legibility Evaluation Pipeline',
        ha='center', va='center', fontsize=21, fontweight='bold')

# ── Input-frames border  ────────────────────────────────────────────────────
border = FancyBboxPatch((0.18, 2.8), 7.15, 3.55,
                         boxstyle='round,pad=0.08',
                         linewidth=2.5, edgecolor='#1a3a8a', facecolor='none')
ax.add_patch(border)
ax.text(3.75, 6.1, 'Input Frames',
        ha='center', va='center', fontsize=16, fontweight='bold', color='#1a3a8a')

# ── Cumulative stacks  ──────────────────────────────────────────────────────
# Front box bottom = pipe_y - box_h/2
y_front = pipe_y - box_h / 2        # = 3.55
stacks = [(1, 1.0, 't = 1 s'), (2, 2.7, 't = 2 s'), (3, 4.4, 't = 3 s')]

for n, xc, tlab in stacks:
    x_front = xc - box_w / 2
    # draw back -> front so front paints on top
    for k in range(n - 1, -1, -1):
        shade = '#dce4f5' if k > 0 else '#b0c4e8'
        bx = x_front + k * ox
        by = y_front + k * oy
        box = FancyBboxPatch((bx, by), box_w, box_h,
                              boxstyle='round,pad=0.04',
                              linewidth=1.8, edgecolor='#1a3a8a', facecolor=shade)
        ax.add_patch(box)
    # label inside front box
    ax.text(xc + (n - 1) * ox / 2, pipe_y + (n - 1) * oy / 2, f'Frame {n}',
            ha='center', va='center', fontsize=13, fontweight='bold', color='#1a3a8a')
    # time label just below the front box
    ax.text(xc, y_front - 0.22, tlab,
            ha='center', va='top', fontsize=17, fontweight='bold', color='#111111')

# dots
ax.text(6.15, pipe_y, '. . .', ha='center', va='center',
        fontsize=24, fontweight='bold', color='#1a3a8a')
ax.text(6.15, y_front - 0.18, '. . .', ha='center', va='top',
        fontsize=17, fontweight='bold', color='#555555')

# ── Arrow 1: border -> VLM  (label ABOVE) ───────────────────────────────────
ax.annotate('', xy=(10.2, pipe_y), xytext=(7.45, pipe_y),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=2.2, mutation_scale=24))
ax.text(8.82, pipe_y + 0.45, 'Prefix frames',
        ha='center', va='bottom', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', lw=1.8))

# ── VLM box  (light sky blue, black text) ──────────────────────────
vlm_box = FancyBboxPatch((10.2, 3.4), 2.9, 2.2,
                          boxstyle='round,pad=0.1',
                          linewidth=2.4, edgecolor='#0096c7', facecolor='#ade8f4')
ax.add_patch(vlm_box)
ax.text(11.65, pipe_y, 'Vision-Language\nModel', ha='center', va='center',
        fontsize=16, fontweight='bold', color='#000000', linespacing=1.6)

# ── Arrow 2: VLM -> Output  (label ABOVE) ───────────────────────────────────
ax.annotate('', xy=(15.4, pipe_y), xytext=(13.2, pipe_y),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=2.2, mutation_scale=24))
ax.text(14.3, pipe_y + 0.52, 'Goal inference',
        ha='center', va='bottom', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', lw=1.8))

# ── Output box  (light gold, black text — same style as VLM) ──────────────────
out_box = FancyBboxPatch((15.4, 3.4), 4.4, 2.2,
                          boxstyle='round,pad=0.1',
                          linewidth=2.4, edgecolor='#c77b00', facecolor='#ffd166')
ax.add_patch(out_box)
ax.text(17.6, pipe_y + 0.42, 'Predicted Goal: A',
        ha='center', va='center', fontsize=17, fontweight='bold', color='black')
ax.text(17.6, pipe_y - 0.38, 'p(A) = 0.82,   p(B) = 0.18',
        ha='center', va='center', fontsize=15, fontweight='bold', color='black')

# ── Probability definitions (below output box: P(A), P(B), then where) ──────
ox_def = 15.2
ax.text(ox_def, 2.85, 'P(A)  =  Probability of picking goal  A',
        ha='left', va='center', fontsize=15, fontweight='bold')
ax.text(ox_def, 2.1, 'P(B)  =  Probability of picking goal  B',
        ha='left', va='center', fontsize=15, fontweight='bold')
ax.text(ox_def, 1.35, 'where :  P(A) + P(B) = 1',
        ha='left', va='center', fontsize=14, style='italic', fontweight='bold')

plt.tight_layout()
out_path = 'C:/Users/anude/OneDrive/Documents/gemini_vlm_eval/analysis_results_2/vlm_flow_simplified.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print('Saved to ' + out_path)
plt.close()
