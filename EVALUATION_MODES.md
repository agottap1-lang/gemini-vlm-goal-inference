# Evaluation Modes Implementation Summary

## Overview

Added **two evaluation modes** to the gemini_vlm_eval framework:

1. **`single_frame`** (default): Baseline mode evaluating each timestamp independently with NO temporal context
2. **`prefix_frames`**: Temporal context mode where each timestamp receives ALL prior frames

## Key Changes

### 1. Core Components Modified

#### `src/gemini_vlm_eval/video.py`
- Added `extract_and_cache_frames()` function
- Extracts frames at integer seconds and returns dict mapping t_sec → jpeg_bytes
- Optional frame caching to `outputs/frames/{video_id}/t_{t_sec:03d}.png`

#### `src/gemini_vlm_eval/prompt.py`
- Added `mode` parameter to `get_instruction_prompt()`
- **single_frame**: "You are given ONLY ONE image... Use ONLY this frame. Do NOT assume you saw earlier or later frames."
- **prefix_frames**: "You are given MULTIPLE images showing frames from t=0 to t=T... Frames are ordered from earliest to latest; you have observed the motion up to time t=T s."

#### `src/gemini_vlm_eval/client.py`
- Updated `evaluate_frame()` to accept `Union[bytes, List[bytes]]`
- Added `mode` parameter
- **single_frame**: Sends one image
- **prefix_frames**: Sends list of images in chronological order
- Tracks `evaluation_mode` in API metadata

#### `src/gemini_vlm_eval/schema.py`
- Added `evaluation_mode: str` field to `EvaluationResult`
- Default: "single_frame"
- Records which mode was used for each evaluation

### 2. Scripts Updated

#### `scripts/eval_dataset.py` (Canonical Entrypoint)
- Added `--mode {single_frame, prefix_frames}` flag (default: single_frame)
- Added `--save-frames` flag to export sampled frames
- Updated `evaluate_video_for_k_seconds()` to:
  - Extract and cache all frames first
  - Collect prefix frames [0..t] for prefix_frames mode
  - Pass single frame for single_frame mode
- Stores `evaluation_mode` and `save_frames` in `run_info.json`

#### `scripts/evaluate_video.py`
- Added `MODE` and `SAVE_FRAMES` configuration variables
- Updated to use new frame extraction and mode-aware evaluation

### 3. Documentation

#### `README.md`
- **New Section**: "Evaluation Modes Explained" with comparison table
- **Updated Usage**: Step 3 now shows both modes with examples
- **New Section**: "Advanced Usage" with mode comparison workflow
- **Updated Anti-Leakage**: Documents what model sees in each mode
- Added `--save-frames` flag documentation

## Usage

### Single-Frame Mode (Default)
```bash
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --out outputs/results_single_frame.jsonl
```

### Prefix-Frames Mode
```bash
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --mode prefix_frames \
  --out outputs/results_prefix_frames.jsonl
```

### With Frame Caching
```bash
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k 10 \
  --mode prefix_frames \
  --save-frames \
  --out outputs/results_with_frames.jsonl
```

Frames saved to: `outputs/frames/{video_id}/t_{t_sec:03d}.png`

## Comparison

| Aspect | `single_frame` | `prefix_frames` |
|--------|----------------|-----------------|
| **Images sent** | 1 frame at t | All frames 0..t |
| **Temporal context** | None (memoryless) | Cumulative (full history) |
| **API cost** | Lower (1 image/call) | Higher (N images at t=N) |
| **Use case** | Snapshot legibility | Motion-based legibility |
| **Research question** | Can single frames reveal intent? | Does observing motion help? |

## Implementation Details

### Frame Extraction
- All frames extracted and cached in memory first (dict: t_sec → bytes)
- Avoids repeated video decoding for prefix_frames mode
- Optional disk caching with `--save-frames` for inspection

### Prefix Frame Collection
At each timestamp t, for prefix_frames mode:
```python
prefix_frames = [frames_dict[t] for t in range(0, t_sec + 1) if t in frames_dict]
```
Results in list: [frame_0, frame_1, ..., frame_t]

### API Call
- **single_frame**: `contents = [prompt, image]`
- **prefix_frames**: `contents = [prompt, image_0, image_1, ..., image_t]`

### Metadata Tracking
Every evaluation result includes:
- `evaluation_mode`: "single_frame" or "prefix_frames"
- Run-level metadata in `run_info.json` includes mode and save_frames flag

## Testing

Run comprehensive test suite:
```bash
python scripts/test_evaluation_modes.py
```

Tests verify:
- ✓ Single-frame prompt generation
- ✓ Prefix-frames prompt generation
- ✓ Schema evaluation_mode field
- ✓ Frame extraction and caching
- ✓ Prefix frame collection logic

## Backward Compatibility

- Default mode is `single_frame` (preserves existing behavior)
- Existing scripts work without changes
- Schema includes new field but maintains all existing fields
- Old JSONL files can still be analyzed (evaluation_mode defaults to "single_frame")

## Files Changed

### Created
- `scripts/test_evaluation_modes.py` - Test suite for new modes
- `EVALUATION_MODES.md` - This document

### Modified
- `src/gemini_vlm_eval/video.py` - Added extract_and_cache_frames()
- `src/gemini_vlm_eval/prompt.py` - Added mode parameter
- `src/gemini_vlm_eval/client.py` - Multi-image support + mode tracking
- `src/gemini_vlm_eval/schema.py` - Added evaluation_mode field
- `scripts/eval_dataset.py` - Added --mode and --save-frames flags
- `scripts/evaluate_video.py` - Added MODE and SAVE_FRAMES config
- `README.md` - Comprehensive documentation of both modes

## Research Use Cases

### Baseline Study
Use `single_frame` to establish baseline legibility judgments from static snapshots.

### Temporal Context Study
Use `prefix_frames` to test whether cumulative motion observation improves legibility detection.

### Comparative Analysis
Run both modes and compare:
- When does prefix_frames improve confidence?
- Are there videos where single_frame is sufficient?
- How does legibility emerge over time in prefix_frames mode?

### Frame Inspection
Use `--save-frames` to:
- Verify exactly what frames were sent to the model
- Debug unexpected results
- Create figures for publications
- Analyze temporal sampling quality

## Next Steps

For users wanting to:

1. **Run baseline evaluation**: Use default `single_frame` mode
2. **Study temporal effects**: Add `--mode prefix_frames`
3. **Compare both approaches**: Run both modes and analyze differences
4. **Inspect frames**: Add `--save-frames` to export frames for verification
5. **Reproduce results**: Check `evaluation_mode` field in JSONL and `run_info.json`
