import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# --- Colour palette (light, academic) ---
BG        = '#F7F9FC'   # page background
CARD_BG   = '#FFFFFF'   # code card
CARD_EDGE = '#C8D6E5'
TITLE_COL = '#1A2744'
KW_COL    = '#2563EB'   # keywords  (blue)
BRANCH_COL= '#7C3AED'   # IF / ELSE (purple)
STR_COL   = '#15803D'   # strings   (green)
INSTR_COL = '#B45309'   # instruction lines (amber)
CMT_COL   = '#6B7280'   # comments  (grey)
RET_COL   = '#DC2626'   # RETURN    (red)
NORM_COL  = '#111827'

fig, ax = plt.subplots(figsize=(16, 8.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Card background
card = FancyBboxPatch((0.3, 0.6), 11.4, 9.0, boxstyle='round,pad=0.18',
                      linewidth=1.8, edgecolor=CARD_EDGE, facecolor=CARD_BG,
                      zorder=0, alpha=0.97)
ax.add_patch(card)

# Thin left accent bar
accent = FancyBboxPatch((0.3, 0.6), 0.18, 9.0, boxstyle='round,pad=0.0',
                        linewidth=0, facecolor='#2563EB', zorder=1, alpha=0.85)
ax.add_patch(accent)

# Title
# (x, y, color, text, size, weight, style)
lines = [
    (0.7, 8.4,  KW_COL,    'FUNCTION  build_prompt(goal_A, goal_B, timestamp, mode):', 22, 'bold',   'normal'),
    (1.1, 7.75, BRANCH_COL,'  IF  mode == "prefix_frames" :',                          21, 'bold',   'normal'),
    (1.7, 7.2,  STR_COL,   '      context  ←  "Frames t = 0 … t = T  provided"',      20, 'normal', 'normal'),
    (1.1, 6.6,  BRANCH_COL,'  ELSE  (single_frame) :',                                 21, 'bold',   'normal'),
    (1.7, 6.05, STR_COL,   '      context  ←  "Single frame at t = T  provided"',      20, 'normal', 'normal'),
    (0.7, 5.35, CMT_COL,   '  # Assemble reasoning instructions',                      18, 'normal', 'italic'),
    (1.1, 4.8,  NORM_COL,  '  prompt  ←  context',                                    20, 'normal', 'normal'),
    (1.7, 4.25, INSTR_COL, '      +  "Goal A: {goal_A}   |   Goal B: {goal_B}"',       20, 'normal', 'normal'),
    (1.7, 3.7,  INSTR_COL, '      +  "Observe LATERAL BIAS of arm path"',              20, 'normal', 'normal'),
    (1.7, 3.15, INSTR_COL, '      +  "Assign pA, pB   s.t.   pA + pB = 1"',           20, 'normal', 'normal'),
    (1.7, 2.6,  INSTR_COL, '      +  "Label legibility:  legible_now | not_yet"',      20, 'normal', 'normal'),
    (1.7, 2.05, INSTR_COL, '      +  "Return ONLY valid JSON: { pA, pB, cue, legible }"', 20, 'normal', 'normal'),
    (0.7, 1.35, RET_COL,   '  RETURN  prompt',                                         21, 'bold',   'normal'),
]

for (x, y, color, text, size, weight, style) in lines:
    ax.text(x, y, text, fontsize=size, color=color, va='center',
            fontfamily='monospace', fontweight=weight,
            style='italic' if style == 'italic' else 'normal')

plt.tight_layout(pad=0.4)
plt.savefig('analysis_results_2/prompt_pseudocode.png', dpi=150, bbox_inches='tight', facecolor=BG)
print('Saved analysis_results_2/prompt_pseudocode.png')
