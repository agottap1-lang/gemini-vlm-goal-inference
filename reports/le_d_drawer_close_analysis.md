**Video:** le_d_drawer_close
**Path:** videos/le d drawer close.mp4
**Ground Truth:** Goal A (close the left drawer)
**Trajectory Type:** legible
**Duration:** ~12 seconds (1fps sampling)

**Statistics:**
- Predicted A: 1 frames
- Predicted B: 4 frames
- Uncertain (C): 7 frames
- Legible moments: 5 frames
- Avg confidence: 69.1%
- Max confidence: 99%
- Min confidence: 50%

**High-Confidence Moments (>= 70%):**
- t=2s: choice=B, confidence=95% -> Robot is oriented towards the shelving unit on its
- t=6s: choice=B, confidence=95% -> the drawer unit is positioned to the right of the 
- t=7s: choice=B, confidence=95% -> The only visible drawer unit is to the right of th
- t=9s: choice=A, confidence=99% -> robot's end-effector is positioned in front of the
- t=10s: choice=B, confidence=95% -> Robot's gripper is positioned in front of the righ

**Interpretation:**
Legible trajectory toward goal B detected with high confidence peaks.

| t_sec | pA/pB | choice | confidence | legible | cue |
|-------|-------|--------|------------|---------|-----|
| 0 | 0.50/0.50 | C | 50 | not_legible_yet | No distinct 'left' or 'right' drawers are visible, |
| 1 | 0.50/0.50 | C | 50 | not_legible_yet | Robot gripper is not aligned with any specific dra |
| 2 | 0.05/0.95 | B | 95 | legible_now | Robot is oriented towards the shelving unit on its |
| 3 | 0.50/0.50 | C | 50 | not_legible_yet | No distinct 'left' or 'right' drawers are visible  |
| 4 | 0.50/0.50 | C | 50 | not_legible_yet | Robot is in a neutral pose, not interacting with a |
| 5 | 0.50/0.50 | C | 50 | not_legible_yet | Robot's gripper is not near any drawer and no left |
| 6 | 0.05/0.95 | B | 95 | legible_now | the drawer unit is positioned to the right of the  |
| 7 | 0.05/0.95 | B | 95 | legible_now | The only visible drawer unit is to the right of th |
| 8 | 0.50/0.50 | C | 50 | not_legible_yet | Only one drawer unit is visible, offering no visua |
| 9 | 0.99/0.01 | A | 99 | legible_now | robot's end-effector is positioned in front of the |
| 10 | 0.05/0.95 | B | 95 | legible_now | Robot's gripper is positioned in front of the righ |
| 11 | 0.50/0.50 | C | 50 | not_legible_yet | Only one drawer unit is visible, making it impossi |