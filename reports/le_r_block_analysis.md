**Video:** le_r_block
**Path:** videos/le r block.mp4
**Ground Truth:** Goal B (pick the right block)
**Trajectory Type:** legible
**Duration:** ~12 seconds (1fps sampling)

**Statistics:**
- Predicted A: 0 frames
- Predicted B: 3 frames
- Uncertain (C): 9 frames
- Legible moments: 3 frames
- Avg confidence: 61.6%
- Max confidence: 95%
- Min confidence: 50%

**High-Confidence Moments (>= 70%):**
- t=7s: choice=B, confidence=95% -> gripper positioned directly above the right block
- t=10s: choice=B, confidence=95% -> gripper is centered above the right block and illu
- t=11s: choice=B, confidence=95% -> gripper is directly above the right block

**Interpretation:**
Legible trajectory toward goal B detected with high confidence peaks.

| t_sec | pA/pB | choice | confidence | legible | cue |
|-------|-------|--------|------------|---------|-----|
| 0 | 0.50/0.50 | C | 50 | not_legible_yet | robot arm is centered and high, equidistant from b |
| 1 | 0.51/0.49 | C | 51 | not_legible_yet | gripper's horizontal position slightly to the left |
| 2 | 0.50/0.50 | C | 50 | not_legible_yet | robot arm is centrally positioned above the space  |
| 3 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is centered horizontally between both bloc |
| 4 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is centered and equidistant from both bloc |
| 5 | 0.50/0.50 | C | 50 | not_legible_yet | gripper positioned centrally above both blocks |
| 6 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is equidistant and centered above both blo |
| 7 | 0.05/0.95 | B | 95 | legible_now | gripper positioned directly above the right block |
| 8 | 0.50/0.50 | C | 50 | not_legible_yet | end effector centered between blocks |
| 9 | 0.53/0.47 | C | 53 | not_legible_yet | gripper positioned slightly left of center between |
| 10 | 0.05/0.95 | B | 95 | legible_now | gripper is centered above the right block and illu |
| 11 | 0.05/0.95 | B | 95 | legible_now | gripper is directly above the right block |