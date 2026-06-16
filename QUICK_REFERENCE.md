# Quick Reference: Evaluation Modes

## TL;DR

Two modes now available:
- **`single_frame`** (default): Each timestamp evaluated independently, one frame only
- **`prefix_frames`**: Each timestamp gets all frames from start to current time

## Quick Start

### Run baseline evaluation (single-frame)
```bash
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --out outputs/results.jsonl
```

### Run temporal context evaluation (prefix-frames)
```bash
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --mode prefix_frames \
  --out outputs/results_temporal.jsonl
```

### Export frames for inspection
```bash
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k 10 \
  --mode prefix_frames \
  --save-frames \
  --out outputs/results.jsonl
```

Frames saved to: `outputs/frames/{video_id}/t_{t_sec:03d}.png`

## Mode Comparison

| Aspect | single_frame | prefix_frames |
|--------|--------------|---------------|
| Images sent at t=5 | 1 (frame at t=5) | 6 (frames 0,1,2,3,4,5) |
| Temporal context | None | Full history |
| API cost (10s video) | 10 calls, 10 images | 10 calls, 55 images |
| Cost ratio | 1× | ~5.5× |
| Use case | Snapshot legibility | Motion-based legibility |

## When to Use Each

**Use single_frame for:**
- Baseline evaluation
- Testing static frame understanding
- Lower-cost exploration
- Comparing against human snapshot judgments

**Use prefix_frames for:**
- Motion-aware legibility
- Testing role of temporal dynamics
- Finding when motion becomes distinguishable
- Comparing against human motion observation

## Flags

### eval_dataset.py
```bash
--mode {single_frame,prefix_frames}  # Evaluation mode (default: single_frame)
--save-frames                        # Export frames to disk (optional)
```

### evaluate_video.py
Edit configuration at top of file:
```python
MODE = "prefix_frames"  # or "single_frame"
SAVE_FRAMES = True      # or False
```

## Output

### JSONL Fields
All results include:
- `evaluation_mode`: "single_frame" or "prefix_frames"
- All standard fields: pA, pB, choice, confidence, cue, legible, etc.

### Run Metadata
`run_info_<uuid>.json` includes:
- `evaluation_mode`: Mode used for this run
- `save_frames`: Whether frames were exported
- All standard metadata: git commit, timestamps, versions, etc.

### Cached Frames (if --save-frames)
```
outputs/frames/
  video_id_1/
    t_000.png  # Frame at t=0
    t_001.png  # Frame at t=1
    ...
  video_id_2/
    t_000.png
    ...
```

## Testing

Verify implementation:
```bash
python scripts/test_evaluation_modes.py
```

Should print:
```
ALL TESTS PASSED ✓
```

## Cost Calculator

For a video with N seconds:
- **single_frame**: N API calls with 1 image each = N images
- **prefix_frames**: N API calls with 1,2,3,...,N images = N(N+1)/2 images

Examples:
- 5 seconds: 5 vs 15 images (3× cost)
- 10 seconds: 10 vs 55 images (5.5× cost)
- 20 seconds: 20 vs 210 images (10.5× cost)

**Tip**: Start with `--k 5` for prefix_frames testing

## Common Workflows

### Compare Both Modes
```bash
# Run both
python scripts/eval_dataset.py --manifest data/manifest.jsonl --k all --mode single_frame --out outputs/single.jsonl
python scripts/eval_dataset.py --manifest data/manifest.jsonl --k all --mode prefix_frames --out outputs/prefix.jsonl

# Generate reports
python scripts/analyze_jsonl.py outputs/single.jsonl --output reports/single.md
python scripts/analyze_jsonl.py outputs/prefix.jsonl --output reports/prefix.md

# Compare results manually or with custom script
```

### Verify Frame Quality
```bash
# Export frames
python scripts/eval_dataset.py --manifest data/manifest.jsonl --k 5 --save-frames --out outputs/test.jsonl

# Inspect outputs/frames/{video_id}/ directory
# Verify frames match expected timestamps
```

### Production Run
```bash
# Validate manifest first
python scripts/validate_manifest.py --manifest data/manifest.jsonl

# Run evaluation
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --mode prefix_frames \
  --out outputs/final_results.jsonl

# Generate analysis
python scripts/analyze_jsonl.py \
  outputs/final_results.jsonl \
  --output reports/final_analysis.md
```

## Troubleshooting

### "No such file or directory: outputs/frames/..."
- Frames only saved if `--save-frames` flag used
- Check that output directory is writable

### API rate limits
- prefix_frames uses more API calls due to multiple images
- Reduce k or add delays between videos if needed

### Memory issues
- Frame caching stores all frames in memory
- For very long videos, consider processing in chunks

## Documentation

- **README.md**: Complete user guide
- **EVALUATION_MODES.md**: Implementation details
- **CHANGELOG.md**: All changes documented
- **example_workflow.py**: Research use cases

## Support

Run tests to verify installation:
```bash
python scripts/test_evaluation_modes.py
```

Check pipeline audit for full details:
```bash
cat reports/pipeline_audit.md
```
