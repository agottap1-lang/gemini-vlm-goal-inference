# Presenter Notes: Manifest, Prompt, and Formulas

## What the manifest does
- `data/manifest.jsonl` is the dataset contract: one JSON row defines one video.
- It stores `video_path`, `goal_A`, `goal_B`, `goal_gt`, `traj_type`, and metadata like `scene_id` and `task_family`.
- `goal_gt` and `traj_type` are not sent to the VLM. They are used later for correctness and grouped analysis.

## What the prompt does
- `src/gemini_vlm_eval/prompt.py` builds the text prompt from `goal_A`, `goal_B`, `t_sec`, `video_id`, and `mode`.
- The model is asked to return only four things: `pA`, `pB`, `cue`, and `legible`.
- In `prefix_frames` mode, the model sees all sampled frames from `t=0` to the current `t`.
- In `single_frame` mode, the model sees only one frame at the current timestamp.

## Exact value provenance
- From the VLM: `pA`, `pB`, `cue`, `legible`
- From hardcoded code: `choice`, `confidence`
- From the manifest: `goal_gt`, `traj_type`, `goal_A`, `goal_B`, `video_id`, `video_path`

## Exact formulas used in code
- `max_p = max(pA, pB)`
- `confidence = int(round(max_p * 100))`
- `choice = 'A'` if `max_p >= 0.52` and `pA >= pB`
- `choice = 'B'` if `max_p >= 0.52` and `pB > pA`
- `choice = 'C'` if `max_p < 0.52`
- `correct = 1 if choice == goal_gt else 0`
- `L_vlm = {t : legible(t) == 'legible_now'}`
- `TTL_vlm = min(L_vlm)` if that set is non-empty, else `None`
- `IoU = |L_vlm INTERSECT L_human| / |L_vlm UNION L_human|`

## Important clarifications for your presentation
- `legible` is not computed from `pA` and `pB` in the code. It is directly predicted by the VLM.
- `choice` can be `C` even though the prompt encourages decisive probabilities, because the code keeps a small uncertainty band below `0.52`.
- The current prompt template is reusable for new two-goal videos in the same benchmark setup, but it is not fully universal for arbitrary tasks because the interpretation logic is still hand-written.
- Some documentation files still say the threshold is `0.60`, but the live code in `client.py` and `schema.py` uses `0.52`.

## Good presenter one-liner
The manifest tells the pipeline what the two goals and ground truth are, the prompt asks the VLM for probabilities and a cue, and the code deterministically converts those probabilities into `choice` and `confidence` before comparing them against the manifest ground truth.
