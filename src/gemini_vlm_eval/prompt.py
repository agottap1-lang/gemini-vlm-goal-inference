def get_instruction_prompt(goal_A: str, goal_B: str, t_sec: int, video_id: str, mode: str = "single_frame") -> str:
    """
    Generate evaluation prompt.
    
    Args:
        goal_A: Description of goal A
        goal_B: Description of goal B
        t_sec: Current timestamp in seconds
        video_id: Video identifier
        mode: Evaluation mode - "single_frame" or "prefix_frames"
    
    Returns:
        Prompt string
    """
    
    # Context depends on evaluation mode
    if mode == "prefix_frames":
        context_text = f"""
You are evaluating LEGIBILITY of a robot arm trajectory: how early a typical human observer can infer the robot's intended goal from its path shape.

You are given MULTIPLE images showing frames from t=0 to t={t_sec} seconds from video_id = "{video_id}".
Frames are ordered from earliest to latest; you have observed the motion up to time t={t_sec}s.
Use ALL frames provided to analyze the robot arm's APPROACH PATH."""
    else:  # single_frame (default)
        context_text = f"""
You are evaluating LEGIBILITY of a robot arm trajectory: how early a typical human observer can infer the robot's intended goal from its path shape.

You are given ONLY ONE image: a single video frame captured at time t = {t_sec} seconds from video_id = "{video_id}".
Use ONLY this frame. Do NOT assume you saw earlier or later frames."""
    
    return f"""{context_text}

There are exactly two candidate goals:
Goal A: {goal_A}
Goal B: {goal_B}

TASK-SPECIFIC CONTEXT: In this robot pick task, the robot approaches its target block by curving its arm path laterally. A robot committed to the LEFT goal will show a path that curves LEFT of center during the approach. A robot committed to the RIGHT goal will show a path that curves RIGHT of center. The key signal is the LATERAL BIAS of the arm's trajectory arc, NOT just the final gripper position.

Look at the arm's PATH SHAPE across the frames (if multiple frames are provided): does the arm's trajectory bow LEFT, bow RIGHT, or remain straight/ambiguous? Use this lateral path curvature as evidence for the goal.

Based on the arm's path shape and any other available evidence, assign probabilities:
- pA = P(Goal A | observed frames)  [a strong left-curving path supports Goal A if Goal A is the left target]
- pB = P(Goal B | observed frames)
Constraints:
- 0 <= pA, pB <= 1
- pA + pB = 1 (within rounding)

YOU MUST MAKE A DECISIVE CALL. If you observe any lateral bias in the arm path, commit to it.
Only output pA=0.5 if there is truly NO directional signal whatsoever (e.g. arm is stationary at center at t=0).

Provide EXACTLY ONE short visual cue describing the arm's PATH SHAPE (e.g. "arm bows left of center", "arm tracks directly right", "arm at home position - no path visible yet").
Also output legibility:
- "legible_now" if the path shape is clear enough that a human could confidently infer the goal NOW
- "not_legible_yet" if the motion is still ambiguous

Output ONLY valid JSON with keys: pA, pB, cue, legible.
No markdown. No extra text. No code fences.
Example format:
{{"pA": 0.78, "pB": 0.22, "cue": "arm arc bows distinctly leftward during approach", "legible": "legible_now"}}
"""

