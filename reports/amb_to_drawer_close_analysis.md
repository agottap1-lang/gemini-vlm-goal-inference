**Video:** amb_to_drawer_close
**Path:** videos/amb to drawer close.mp4
**Ground Truth:** Goal B (close the top drawer)
**Trajectory Type:** ambiguous
**Duration:** ~16 seconds (1fps sampling)

**Statistics:**
- Predicted A: 3 frames
- Predicted B: 7 frames
- Uncertain (C): 6 frames
- Legible moments: 10 frames
- Avg confidence: 73.9%
- Max confidence: 100%
- Min confidence: 50%

**High-Confidence Moments (>= 70%):**
- t=4s: choice=B, confidence=80% -> Robot's end effector height is closer to the top d
- t=5s: choice=A, confidence=80% -> Robot's gripper is vertically aligned closer to th
- t=7s: choice=B, confidence=70% -> Robot's end-effector is positioned significantly a
- t=9s: choice=B, confidence=85% -> robot's gripper is positioned near the top open dr
- t=10s: choice=B, confidence=80% -> gripper's vertical position is closely aligned wit
- t=11s: choice=B, confidence=99% -> robot gripper is positioned directly above the ope
- t=12s: choice=B, confidence=95% -> gripper is positioned vertically above the top dra
- t=13s: choice=B, confidence=95% -> Robot end-effector height is aligned with the top 
- t=14s: choice=A, confidence=99% -> The bottom drawer is open.
- t=15s: choice=A, confidence=100% -> The bottom drawer is open

**Interpretation:**
Ambiguous trajectory confirmed: model frequently uncertain (choice=C). Goal identification is challenging throughout.

| t_sec | pA/pB | choice | confidence | legible | cue |
|-------|-------|--------|------------|---------|-----|
| 0 | 0.50/0.50 | C | 50 | not_legible_yet | Robot gripper is high and not yet aligned with eit |
| 1 | 0.50/0.50 | C | 50 | not_legible_yet | robot arm is in a neutral, raised position, not or |
| 2 | 0.50/0.50 | C | 50 | not_legible_yet | All drawers are currently in a closed state, and t |
| 3 | 0.50/0.50 | C | 50 | not_legible_yet | Robot's end-effector is well above both drawers an |
| 4 | 0.20/0.80 | B | 80 | legible_now | Robot's end effector height is closer to the top d |
| 5 | 0.80/0.20 | A | 80 | legible_now | Robot's gripper is vertically aligned closer to th |
| 6 | 0.50/0.50 | C | 50 | not_legible_yet | Robot's arm is retracted and not near either open  |
| 7 | 0.30/0.70 | B | 70 | legible_now | Robot's end-effector is positioned significantly a |
| 8 | 0.50/0.50 | C | 50 | not_legible_yet | Robot gripper is positioned vertically between the |
| 9 | 0.15/0.85 | B | 85 | legible_now | robot's gripper is positioned near the top open dr |
| 10 | 0.20/0.80 | B | 80 | legible_now | gripper's vertical position is closely aligned wit |
| 11 | 0.01/0.99 | B | 99 | legible_now | robot gripper is positioned directly above the ope |
| 12 | 0.05/0.95 | B | 95 | legible_now | gripper is positioned vertically above the top dra |
| 13 | 0.05/0.95 | B | 95 | legible_now | Robot end-effector height is aligned with the top  |
| 14 | 0.99/0.01 | A | 99 | legible_now | The bottom drawer is open. |
| 15 | 1.00/0.00 | A | 100 | legible_now | The bottom drawer is open |