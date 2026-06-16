# Implementation Complete: Two Evaluation Modes

## Summary

Successfully added a **second evaluation mode** to gemini_vlm_eval while keeping the existing `single_frame` baseline unchanged.

## What Was Added

### Two Evaluation Modes

1. **`single_frame` (default)** - Baseline memoryless evaluation
   - At each timestamp t, send ONLY the frame at t_sec
   - Prompt: "Use ONLY this frame. Do NOT assume you saw earlier or later frames."
   - Use case: Test if snapshots reveal intent

2. **`prefix_frames`** - Temporal context evaluation
   - At each timestamp t, send ALL frames from 0 to t (inclusive)
   - Prompt: "Frames are ordered from earliest to latest; you have observed the motion up to time t=T s."
   - Use case: Test if observing motion helps

### New Features

- `--mode {single_frame, prefix_frames}` flag in eval_dataset.py
- `--save-frames` flag to export frames to `outputs/frames/{video_id}/t_{t_sec:03d}.png`
- Frame extraction and caching to avoid repeated video decoding
- `evaluation_mode` field in JSONL output for tracking which mode was used
- Mode and save_frames flags recorded in `run_info.json` for reproducibility

## Files Modified

### Core Library (`src/gemini_vlm_eval/`)

1. **video.py**
   - Added `extract_and_cache_frames()` function
   - Extracts frames at integer seconds → dict {t_sec: jpeg_bytes}
   - Optional disk caching for frame inspection

2. **prompt.py**
   - Added `mode` parameter to `get_instruction_prompt()`
   - Different prompts for single_frame vs prefix_frames

3. **client.py**
   - Updated `evaluate_frame()` to accept `Union[bytes, List[bytes]]`
   - Single image for single_frame, list of images for prefix_frames
   - Tracks `evaluation_mode` in metadata

4. **schema.py**
   - Added `evaluation_mode: str` field to `EvaluationResult`
   - Default: "single_frame"

### Scripts

5. **eval_dataset.py** (Canonical Entrypoint)
   - Added `--mode` and `--save-frames` flags
   - Frame caching before evaluation
   - Prefix frame collection for prefix_frames mode
   - Mode tracking in run_info.json

6. **evaluate_video.py**
   - Added `MODE` and `SAVE_FRAMES` configuration variables
   - Updated to use new extraction and mode-aware evaluation

### Documentation

7. **README.md**
   - New section: "Evaluation Modes Explained" with comparison table
   - Updated Step 3 with mode examples
   - New section: "Advanced Usage" with comparison workflow
   - Updated Anti-Leakage section to document both modes

8. **EVALUATION_MODES.md** (New)
   - Comprehensive implementation documentation
   - Usage examples for both modes
   - Research use cases
   - Cost considerations

9. **scripts/test_evaluation_modes.py** (New)
   - Test suite verifying prompt generation, schema, extraction
   - All tests pass ✓

10. **scripts/example_workflow.py** (New)
    - Example commands for both modes
    - Research use case examples
    - Cost comparison

## Usage Examples

### Baseline (Single-Frame)
```bash
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --out outputs/results_single_frame.jsonl
```

### Temporal Context (Prefix-Frames)
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

## Backward Compatibility

✓ **Fully backward compatible**
- Default mode is `single_frame` (preserves existing behavior)
- All existing scripts work without changes
- Old JSONL files compatible (evaluation_mode defaults to "single_frame")
- Schema maintains all existing fields

## Testing

Run comprehensive test suite:
```bash
python scripts/test_evaluation_modes.py
```

All tests pass:
- ✓ Single-frame prompt generation
- ✓ Prefix-frames prompt generation  
- ✓ Schema evaluation_mode field
- ✓ Frame extraction and caching
- ✓ Prefix frame collection logic

## Key Implementation Details

### Frame Extraction
- Extract all frames ONCE and cache in memory (dict: t_sec → bytes)
- Avoids repeated video decoding for prefix_frames mode
- Optional disk export with `--save-frames`

### Prefix Frame Collection
At timestamp t in prefix_frames mode:
```python
prefix_frames = [frames_dict[t] for t in range(0, t_sec + 1) if t in frames_dict]
# Result: [frame_0, frame_1, ..., frame_t]
```

### API Call Structure
- **single_frame**: `contents = [prompt, image]`
- **prefix_frames**: `contents = [prompt, image_0, image_1, ..., image_t]`

### Metadata Tracking
Every result includes:
- Per-frame: `evaluation_mode` field in JSONL
- Per-run: `evaluation_mode` and `save_frames` in run_info.json

## Cost Comparison

| Mode | Images per Call | Total for N seconds |
|------|-----------------|---------------------|
| single_frame | 1 | N images |
| prefix_frames | 1, 2, 3, ..., N | N(N+1)/2 images |

**Cost ratio**: prefix_frames is ~N/2 times more expensive
- 10 seconds: 5.5× cost
- 20 seconds: 10.5× cost

**Recommendation**: Start with small k for prefix_frames testing

## Research Applications

1. **Baseline Study**: Use single_frame to establish snapshot-based legibility
2. **Temporal Effect**: Compare both modes to measure motion's contribution
3. **Emergence Study**: Use prefix_frames to find when motion becomes legible
4. **Frame Verification**: Use --save-frames to inspect exact model inputs

## Next Steps for Users

1. **Quick test**: Run both modes on k=5 seconds to compare
2. **Full evaluation**: Run both modes on k=all for complete analysis
3. **Compare results**: Analyze confidence differences and choice agreement
4. **Inspect frames**: Use --save-frames to verify model inputs
5. **Publication**: Use evaluation_mode field to separate analyses

## Documentation

- **README.md**: User-facing documentation with examples
- **EVALUATION_MODES.md**: Implementation details and research use cases
- **example_workflow.py**: Example commands and research workflows
- **test_evaluation_modes.py**: Verification test suite

All documentation is comprehensive and research-ready.

---

**Status**: ✅ IMPLEMENTATION COMPLETE AND TESTED

The repository now supports both single-frame (baseline) and prefix-frames (temporal) evaluation modes, with full documentation, testing, and backward compatibility.
