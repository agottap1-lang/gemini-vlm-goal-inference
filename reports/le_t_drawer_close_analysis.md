**Video:** le_t_drawer_close
**Path:** videos/le t drawer close.mp4
**Ground Truth:** Goal A (close the bottom drawer)
**Trajectory Type:** legible
**Duration:** ~9 seconds (1fps sampling)

**Statistics:**
- Predicted A: 1 frames
- Predicted B: 6 frames
- Uncertain (C): 2 frames
- Legible moments: 7 frames
- Avg confidence: 84.9%
- Max confidence: 100%
- Min confidence: 50%

**High-Confidence Moments (>= 70%):**
- t=0s: choice=B, confidence=99% -> The top drawer is open, while the bottom drawer is
- t=1s: choice=B, confidence=80% -> Robot gripper is positioned above and at the heigh
- t=3s: choice=B, confidence=95% -> Robot end-effector's vertical position is aligned 
- t=5s: choice=B, confidence=95% -> robot's gripper is positioned in front of the open
- t=6s: choice=B, confidence=90% -> robot gripper is positioned at a height closer to 
- t=7s: choice=A, confidence=100% -> The top drawer is already closed.
- t=8s: choice=B, confidence=100% -> The top drawer is open, while the bottom drawer is

**Interpretation:**
Legible trajectory toward goal B detected with high confidence peaks.

| t_sec | pA/pB | choice | confidence | legible | cue |
|-------|-------|--------|------------|---------|-----|
| 0 | 0.01/0.99 | B | 99 | legible_now | The top drawer is open, while the bottom drawer is |
| 1 | 0.20/0.80 | B | 80 | legible_now | Robot gripper is positioned above and at the heigh |
| 2 | 0.45/0.55 | C | 55 | not_legible_yet | Robot's end effector vertical position is slightly |
| 3 | 0.05/0.95 | B | 95 | legible_now | Robot end-effector's vertical position is aligned  |
| 4 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is positioned above the middle drawer |
| 5 | 0.05/0.95 | B | 95 | legible_now | robot's gripper is positioned in front of the open |
| 6 | 0.10/0.90 | B | 90 | legible_now | robot gripper is positioned at a height closer to  |
| 7 | 1.00/0.00 | A | 100 | legible_now | The top drawer is already closed. |
| 8 | 0.00/1.00 | B | 100 | legible_now | The top drawer is open, while the bottom drawer is |