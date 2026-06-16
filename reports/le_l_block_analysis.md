**Video:** le_l_block
**Path:** videos/le l block.mp4
**Ground Truth:** Goal A (pick the left block)
**Trajectory Type:** legible
**Duration:** ~15 seconds (1fps sampling)

**Statistics:**
- Predicted A: 4 frames
- Predicted B: 0 frames
- Uncertain (C): 11 frames
- Legible moments: 4 frames
- Avg confidence: 62.3%
- Max confidence: 99%
- Min confidence: 50%

**High-Confidence Moments (>= 70%):**
- t=5s: choice=A, confidence=95% -> robot end-effector is horizontally aligned with th
- t=7s: choice=A, confidence=95% -> gripper positioned directly above the left block
- t=11s: choice=A, confidence=99% -> gripper is centered directly above the left block 
- t=13s: choice=A, confidence=95% -> gripper positioned directly above the left block

**Interpretation:**
Strong legible trajectory. Model consistently identifies goal A with high confidence during key moments.

| t_sec | pA/pB | choice | confidence | legible | cue |
|-------|-------|--------|------------|---------|-----|
| 0 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is centered high above both blocks |
| 1 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is centered and equidistant from both bloc |
| 2 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is high and centered between blocks |
| 3 | 0.50/0.50 | C | 50 | not_legible_yet | Robot arm is high and centered, equidistant from b |
| 4 | 0.50/0.50 | C | 50 | not_legible_yet | robot gripper is positioned centrally above both b |
| 5 | 0.95/0.05 | A | 95 | legible_now | robot end-effector is horizontally aligned with th |
| 6 | 0.50/0.50 | C | 50 | not_legible_yet | gripper positioned high and centered between the t |
| 7 | 0.95/0.05 | A | 95 | legible_now | gripper positioned directly above the left block |
| 8 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is high and centered, equidistant from bot |
| 9 | 0.50/0.50 | C | 50 | not_legible_yet | Robot arm is centered above its base, holding a cu |
| 10 | 0.50/0.50 | C | 50 | not_legible_yet | gripper positioned equidistant between both blocks |
| 11 | 0.99/0.01 | A | 99 | legible_now | gripper is centered directly above the left block  |
| 12 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is positioned centrally between both block |
| 13 | 0.95/0.05 | A | 95 | legible_now | gripper positioned directly above the left block |
| 14 | 0.50/0.50 | C | 50 | not_legible_yet | gripper is positioned midway between the two block |