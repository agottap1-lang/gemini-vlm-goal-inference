**Video:** amb_r_block
**Path:** videos/amb r block.mp4
**Ground Truth:** Goal B (pick the right block)
**Trajectory Type:** ambiguous
**Duration:** ~15 seconds (1fps sampling)

**Statistics:**
- Predicted A: 0 frames
- Predicted B: 5 frames
- Uncertain (C): 10 frames
- Legible moments: 5 frames
- Avg confidence: 66.0%
- Max confidence: 99%
- Min confidence: 50%

**High-Confidence Moments (>= 70%):**
- t=6s: choice=B, confidence=98% -> gripper is positioned directly above the right blo
- t=11s: choice=B, confidence=95% -> gripper is positioned directly above the right blo
- t=12s: choice=B, confidence=99% -> gripper is positioned directly above the right blo
- t=13s: choice=B, confidence=99% -> gripper is positioned directly above the right blo
- t=14s: choice=B, confidence=99% -> gripper is directly over the right block

**Interpretation:**
Ambiguous trajectory confirmed: model frequently uncertain (choice=C). Goal identification is challenging throughout.

| t_sec | pA/pB | choice | confidence | legible | cue |
|-------|-------|--------|------------|---------|-----|
| 0 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is centered between the two blocks |
| 1 | 0.50/0.50 | C | 50 | not_legible_yet | Robot's gripper is in a high central position, equ |
| 2 | 0.50/0.50 | C | 50 | not_legible_yet | robot's gripper is centered between the two blocks |
| 3 | 0.50/0.50 | C | 50 | not_legible_yet | robot arm is high up and horizontally centered bet |
| 4 | 0.50/0.50 | C | 50 | not_legible_yet | Robot arm is high and centered, not yet oriented t |
| 5 | 0.50/0.50 | C | 50 | not_legible_yet | robot gripper is high above and centered between b |
| 6 | 0.02/0.98 | B | 98 | legible_now | gripper is positioned directly above the right blo |
| 7 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is centered between both blocks |
| 8 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is centered symmetrically between both blo |
| 9 | 0.50/0.50 | C | 50 | not_legible_yet | gripper positioned equidistant between the two blo |
| 10 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is centered horizontally between both bloc |
| 11 | 0.05/0.95 | B | 95 | legible_now | gripper is positioned directly above the right blo |
| 12 | 0.01/0.99 | B | 99 | legible_now | gripper is positioned directly above the right blo |
| 13 | 0.01/0.99 | B | 99 | legible_now | gripper is positioned directly above the right blo |
| 14 | 0.01/0.99 | B | 99 | legible_now | gripper is directly over the right block |