#!/usr/bin/env python3
"""
Example workflow demonstrating both evaluation modes.

This script shows how to:
1. Run single-frame evaluation (baseline)
2. Run prefix-frames evaluation (temporal context)
3. Compare the results

Note: This is an example workflow, not meant to be run directly.
Adapt the commands to your specific needs.
"""

# ============================================================================
# STEP 1: Single-Frame Evaluation (Baseline)
# ============================================================================
# Evaluate each timestamp independently with NO temporal context.
# At t=5, model sees ONLY the frame at 5 seconds.

COMMAND_SINGLE_FRAME = """
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --mode single_frame \
  --out outputs/results_single_frame.jsonl
"""

# Expected output:
# - outputs/results_single_frame.jsonl
# - outputs/run_info_<uuid>.json (includes evaluation_mode: "single_frame")
# - outputs/pip_freeze.txt


# ============================================================================
# STEP 2: Prefix-Frames Evaluation (Temporal Context)
# ============================================================================
# Evaluate each timestamp with ALL prior frames.
# At t=5, model sees frames at t=0,1,2,3,4,5 in chronological order.

COMMAND_PREFIX_FRAMES = """
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --mode prefix_frames \
  --out outputs/results_prefix_frames.jsonl
"""

# Expected output:
# - outputs/results_prefix_frames.jsonl
# - outputs/run_info_<uuid>.json (includes evaluation_mode: "prefix_frames")
# - outputs/pip_freeze.txt


# ============================================================================
# STEP 3: With Frame Caching (Optional)
# ============================================================================
# Save frames to disk for inspection and verification.

COMMAND_WITH_FRAMES = """
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k 10 \
  --mode prefix_frames \
  --save-frames \
  --out outputs/results_with_frames.jsonl
"""

# Expected output:
# - outputs/results_with_frames.jsonl
# - outputs/run_info_<uuid>.json
# - outputs/pip_freeze.txt
# - outputs/frames/{video_id}/t_000.png
# - outputs/frames/{video_id}/t_001.png
# - ...
# - outputs/frames/{video_id}/t_009.png


# ============================================================================
# STEP 4: Generate Analysis Reports
# ============================================================================

COMMAND_ANALYZE_SINGLE = """
python scripts/analyze_jsonl.py \
  outputs/results_single_frame.jsonl \
  --output reports/analysis_single_frame.md
"""

COMMAND_ANALYZE_PREFIX = """
python scripts/analyze_jsonl.py \
  outputs/results_prefix_frames.jsonl \
  --output reports/analysis_prefix_frames.md
"""


# ============================================================================
# STEP 5: Compare Results (Manual Analysis)
# ============================================================================
# Compare the two JSONL files to see how temporal context affects legibility:
#
# Questions to explore:
# 1. Does prefix_frames mode increase confidence scores?
# 2. Are there videos where single_frame is sufficient?
# 3. How does legibility emerge over time in prefix_frames mode?
# 4. At what timestamp does prefix_frames diverge from single_frame?
#
# Example comparison script (pseudo-code):
"""
import json

# Load results
with open('outputs/results_single_frame.jsonl') as f:
    single_results = [json.loads(line) for line in f]

with open('outputs/results_prefix_frames.jsonl') as f:
    prefix_results = [json.loads(line) for line in f]

# Group by video_id and t_sec for comparison
single_by_key = {(r['video_id'], r['t_sec']): r for r in single_results}
prefix_by_key = {(r['video_id'], r['t_sec']): r for r in prefix_results}

# Compare confidence scores
for key in single_by_key:
    if key in prefix_by_key:
        s = single_by_key[key]
        p = prefix_by_key[key]
        
        conf_diff = p['confidence'] - s['confidence']
        choice_same = s['choice'] == p['choice']
        
        print(f"{key}: conf_diff={conf_diff:+3d}, choice_same={choice_same}")
"""


# ============================================================================
# Research Use Cases
# ============================================================================

# USE CASE 1: Baseline Legibility Study
# Question: Can static snapshots reveal motion intent?
# Method: Use single_frame mode only
RESEARCH_1 = """
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --mode single_frame \
  --out outputs/baseline_study.jsonl
"""

# USE CASE 2: Temporal Context Effect
# Question: Does observing motion improve legibility detection?
# Method: Compare single_frame vs prefix_frames
RESEARCH_2 = """
# Run both modes
python scripts/eval_dataset.py --manifest data/manifest.jsonl --k all --mode single_frame --out outputs/baseline.jsonl
python scripts/eval_dataset.py --manifest data/manifest.jsonl --k all --mode prefix_frames --out outputs/temporal.jsonl

# Analyze difference in confidence, choice, and time-to-legibility
"""

# USE CASE 3: Legibility Emergence Study
# Question: When does motion become legible?
# Method: Use prefix_frames mode and track when confidence crosses threshold
RESEARCH_3 = """
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --mode prefix_frames \
  --save-frames \
  --out outputs/emergence_study.jsonl

# Analyze at what t_sec confidence first exceeds 0.95
# Inspect saved frames to identify critical moments
"""

# USE CASE 4: Frame Verification
# Question: What exactly did the model see?
# Method: Use --save-frames to export frames
RESEARCH_4 = """
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k 15 \
  --mode prefix_frames \
  --save-frames \
  --out outputs/verified_frames.jsonl

# Inspect outputs/frames/{video_id}/ to see exact frames sent to model
# Create figures for publication showing frame sequences
"""


# ============================================================================
# Cost Considerations
# ============================================================================

# Single-frame mode cost: N frames × 1 image/frame
# Example: 10-second video → 10 API calls with 1 image each

# Prefix-frames mode cost: Σ(1..N) images
# Example: 10-second video → 10 API calls with 1,2,3,...,10 images = 55 images total
#
# Cost ratio: prefix_frames / single_frame ≈ N/2
# For 10 seconds: 5.5x more expensive
# For 20 seconds: 10.5x more expensive
#
# Recommendation: Start with small k (e.g., --k 5) for prefix_frames testing


if __name__ == "__main__":
    print(__doc__)
    print("\nThis is an example workflow script showing how to use both evaluation modes.")
    print("Copy and paste the commands above into your terminal, or adapt them for your needs.")
    print("\nFor more details, see:")
    print("  - README.md (Step 3: Run Evaluation)")
    print("  - EVALUATION_MODES.md (Detailed implementation)")
    print("  - reports/pipeline_audit.md (Full audit report)")
