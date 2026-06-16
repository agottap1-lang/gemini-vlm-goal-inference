# CHANGELOG: Two Evaluation Modes Implementation

## Added Features

### Evaluation Modes
- **single_frame mode (default)**: Evaluate each timestamp independently with NO temporal context
  - At time t, send ONLY the frame at t_sec
  - Prompt: "Use ONLY this frame. Do NOT assume you saw earlier or later frames."
  - Use case: Test if snapshots reveal intent (baseline)

- **prefix_frames mode**: Evaluate with cumulative visual context
  - At time t, send ALL frames from 0 to t (inclusive) in chronological order
  - Prompt: "Frames are ordered from earliest to latest; you have observed the motion up to time t=T s."
  - Use case: Test if observing motion improves legibility detection

### Frame Caching
- New `--save-frames` flag exports sampled frames to `outputs/frames/{video_id}/t_{t_sec:03d}.png`
- Useful for frame inspection, debugging, and publication figures

### Metadata Tracking
- Added `evaluation_mode` field to EvaluationResult schema
- Run-level metadata includes `evaluation_mode` and `save_frames` flags
- Enables reproducibility and mode-specific analysis

## Modified Files

### Core Library
- `src/gemini_vlm_eval/video.py`: Added `extract_and_cache_frames()` for frame extraction and caching
- `src/gemini_vlm_eval/prompt.py`: Added `mode` parameter with mode-specific prompts
- `src/gemini_vlm_eval/client.py`: Updated to handle single image or list of images, added mode tracking
- `src/gemini_vlm_eval/schema.py`: Added `evaluation_mode` field to EvaluationResult

### Scripts
- `scripts/eval_dataset.py`: Added `--mode` and `--save-frames` flags, frame caching logic
- `scripts/evaluate_video.py`: Added MODE and SAVE_FRAMES configuration variables

### Documentation
- `README.md`: 
  - New section: "Evaluation Modes Explained" with comparison table
  - Updated usage examples showing both modes
  - New section: "Advanced Usage" with comparison workflows
  - Updated Anti-Leakage documentation

## New Files
- `EVALUATION_MODES.md`: Comprehensive implementation documentation
- `IMPLEMENTATION_SUMMARY.md`: Quick summary of changes
- `scripts/test_evaluation_modes.py`: Test suite for new features (all tests pass ✓)
- `scripts/example_workflow.py`: Example commands and research use cases

## API Changes

### New Parameters
- `get_instruction_prompt()`: Added `mode="single_frame"` parameter
- `GeminiClient.evaluate_frame()`: Added `mode="single_frame"` parameter
- `evaluate_frame()` accepts: `Union[bytes, List[bytes]]` for image_bytes
- `evaluate_video()`: Added `mode` and `save_frames` parameters
- `evaluate_video_for_k_seconds()`: Added `mode` and `save_frames` parameters

### Command-Line Flags
- `eval_dataset.py`: `--mode {single_frame,prefix_frames}` (default: single_frame)
- `eval_dataset.py`: `--save-frames` (optional, saves frames to disk)

## Backward Compatibility

✅ **Fully backward compatible**
- Default mode is `single_frame` (existing behavior preserved)
- All existing scripts work without changes
- Old JSONL output compatible (evaluation_mode defaults to "single_frame")
- No breaking changes to API or schema (only additions)

## Testing

Comprehensive test suite added:
```bash
python scripts/test_evaluation_modes.py
```

All tests pass:
- ✓ Single-frame prompt generation
- ✓ Prefix-frames prompt generation
- ✓ Schema evaluation_mode field
- ✓ Frame extraction and caching
- ✓ Prefix frame collection logic

## Usage Examples

### Single-Frame Mode (Default)
```bash
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --out outputs/results_single.jsonl
```

### Prefix-Frames Mode
```bash
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --mode prefix_frames \
  --out outputs/results_prefix.jsonl
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

## Cost Considerations

**Single-frame mode**: N frames × 1 image/frame = N images total
**Prefix-frames mode**: Σ(1..N) = N(N+1)/2 images total

Cost ratio: ~N/2 (e.g., 10 seconds = 5.5× more expensive)

Recommendation: Start with small k (e.g., --k 5) for prefix_frames testing

## Research Applications

1. **Baseline Study**: Use single_frame to establish snapshot legibility
2. **Temporal Effect Study**: Compare single_frame vs prefix_frames
3. **Legibility Emergence**: Track when motion becomes distinguishable
4. **Frame Verification**: Export frames for inspection and publication

## Documentation

All documentation updated and comprehensive:
- README.md: User-facing guide with examples
- EVALUATION_MODES.md: Implementation details
- IMPLEMENTATION_SUMMARY.md: Quick summary
- example_workflow.py: Research use cases
- test_evaluation_modes.py: Verification tests

## Git Commit Message

```
feat: Add two evaluation modes (single_frame + prefix_frames)

- Add prefix_frames mode for temporal context evaluation
- Keep single_frame as default baseline (backward compatible)
- Add --mode and --save-frames flags to eval_dataset.py
- Add frame caching to outputs/frames/{video_id}/
- Add evaluation_mode field to schema for tracking
- Update prompts to be mode-aware
- Add comprehensive tests (all pass)
- Update README with mode comparison and examples

Closes #[issue-number]
```

## Version

This represents a **minor version bump** (e.g., 0.1.0 → 0.2.0):
- New features added
- Fully backward compatible
- No breaking changes
- Comprehensive testing and documentation
