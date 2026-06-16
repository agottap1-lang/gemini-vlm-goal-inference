**Video:** amb_d_drawer_cclose
**Path:** videos/amb d drawer cclose.mp4
**Ground Truth:** Goal B (close the right drawer)
**Trajectory Type:** ambiguous
**Duration:** ~20 seconds (1fps sampling)

**Statistics:**
- Predicted A: 2 frames
- Predicted B: 2 frames
- Uncertain (C): 16 frames
- Legible moments: 3 frames
- Avg confidence: 59.4%
- Max confidence: 98%
- Min confidence: 50%

**High-Confidence Moments (>= 70%):**
- t=13s: choice=A, confidence=98% -> robot gripper is aligned with the left side of the
- t=16s: choice=B, confidence=95% -> Robot's end-effector is aligned with the drawer un
- t=17s: choice=A, confidence=95% -> robot's end-effector is positioned to the left sid
- t=18s: choice=B, confidence=95% -> robot is positioned to close the drawer unit locat

**Interpretation:**
Ambiguous trajectory confirmed: model frequently uncertain (choice=C). Goal identification is challenging throughout.

| t_sec | pA/pB | choice | confidence | legible | cue |
|-------|-------|--------|------------|---------|-----|
| 0 | 0.50/0.50 | C | 50 | not_legible_yet | No distinct 'left' or 'right' drawers are visible  |
| 1 | 0.50/0.50 | C | 50 | not_legible_yet | robot's end-effector is aligned with the topmost o |
| 2 | 0.55/0.45 | C | 55 | not_legible_yet | Robot's end-effector is positioned directly above  |
| 3 | 0.50/0.50 | C | 50 | not_legible_yet | All drawers are currently closed, and the robot is |
| 4 | 0.50/0.50 | C | 50 | not_legible_yet | The scene displays a single vertical stack of draw |
| 5 | 0.50/0.50 | C | 50 | not_legible_yet | No distinct 'left' or 'right' drawers are visible  |
| 6 | 0.50/0.50 | C | 50 | not_legible_yet | All visible drawers are already closed, and there  |
| 7 | 0.50/0.50 | C | 50 | not_legible_yet | Robot's gripper is holding a cup and is far from t |
| 8 | 0.50/0.50 | C | 50 | not_legible_yet | Robot is holding an object and its arm is not alig |
| 9 | 0.50/0.50 | C | 50 | not_legible_yet | There is a single column of drawers, not distinct  |
| 10 | 0.50/0.50 | C | 50 | not_legible_yet | Robot arm is retracted, and no distinct 'left' or  |
| 11 | 0.50/0.50 | C | 50 | not_legible_yet | No visual distinction between 'left' and 'right' d |
| 12 | 0.50/0.50 | C | 50 | not_legible_yet | The image displays a single stack of drawers, with |
| 13 | 0.98/0.02 | A | 98 | legible_now | robot gripper is aligned with the left side of the |
| 14 | 0.50/0.50 | C | 50 | not_legible_yet | No visual distinction in the frame to label a draw |
| 15 | 0.50/0.50 | C | 50 | not_legible_yet | Robot's gripper is not yet interacting with or cle |
| 16 | 0.05/0.95 | B | 95 | legible_now | Robot's end-effector is aligned with the drawer un |
| 17 | 0.95/0.05 | A | 95 | not_legible_yet | robot's end-effector is positioned to the left sid |
| 18 | 0.05/0.95 | B | 95 | legible_now | robot is positioned to close the drawer unit locat |
| 19 | 0.50/0.50 | C | 50 | not_legible_yet | The visible drawer unit does not have distinct lef |