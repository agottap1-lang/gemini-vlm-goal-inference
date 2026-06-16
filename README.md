# Gemini VLM Evaluation for Robot Motion Legibility

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive framework for evaluating the legibility of robot (and human) motion using Google's Gemini Vision-Language Model (VLM). This tool assesses whether an observer can infer an actor's intended goal from partial video observations, supporting research in human-robot interaction and motion planning.

## Research Context

This project addresses the challenge of making robot motions more legible to human collaborators. By leveraging large vision-language models, we can:

- **Evaluate Legibility**: Determine if robot actions are interpretable from visual cues alone
- **Benchmark Performance**: Compare VLM judgments against human annotations
- **Guide Policy Learning**: Use legibility feedback to improve robot motion planning
- **Study Human-Robot Interaction**: Analyze how legibility affects collaboration

### Key Research Questions
- How well can VLMs judge motion legibility from partial observations?
- Can VLM-based critics improve robot legibility during policy rollout?
- What visual cues contribute most to legibility judgments?

## Features

- **Second-by-Second Evaluation**: Process videos with configurable sampling rates
- **Batch Processing**: Evaluate entire datasets of videos automatically
- **Comprehensive Metrics**: IoU, time-to-legibility, and detailed statistics
- **Flexible Output**: JSONL format with rich metadata for downstream analysis
- **Research-Ready**: Designed for academic studies with human annotation integration
- **Extensible Architecture**: Modular design for custom prompts and evaluation criteria

## Decision Logic & Reproducibility

- **Sampling**: Extract frames at 1.0s intervals (configurable) so every timestamp is treated uniformly.
- **Prompting**: Task-agnostic legibility prompt includes `video_id` and `t_sec`, asks for `pA`, `pB`, `cue`, `legible`.
- **Decision rule**: Let $m=\max(pA,pB)$. If $m \ge 0.60$, choose `A` when $pA \ge pB` else `B`; otherwise choose `C` (uncertain). Confidence is `int(round(m*100))`.
- **Validation**: Pydantic only fills `choice`/`confidence` if missing; otherwise respects provided values.
- **Metadata per frame**: Stored in JSONL: probabilities, choice, confidence, cue, legible flag, frame/time indices, plus API call metadata (provider/model/endpoint, generation params, latency_ms, http_status, retry_count, request_id/response_id, api_key_source="env:GEMINI_API_KEY").
- **Run-level provenance**: Each run writes `outputs/run_info.json` with `run_id`, start/end timestamps (UTC), git commit + dirty flag, python/OS/platform info, OpenCV + google-genai versions, CLI used, `api_key_source`, and `pip_freeze_path`; a pip snapshot is saved to `outputs/pip_freeze.txt`. Machine info includes CPU/core count and RAM if available.

## Installation

### Prerequisites
- Python 3.8 or higher
- Google Cloud API key with Gemini API access

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/gemini-vlm-eval.git
   cd gemini-vlm-eval
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

4. **Set up API credentials:**

   Create a `.env` file in the project root with your API keys:
   ```dotenv
   # Google Gemini (required)
   GEMINI_API_KEY=your-gemini-api-key

   # OpenAI (required for GPT-4o evaluation)
   OPENAI_API_KEY=your-openai-api-key

   # Anthropic / Claude (required for Claude evaluation)
   ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

   Or export them as environment variables:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

   On Windows (PowerShell):
   ```powershell
   $env:GEMINI_API_KEY="your-gemini-api-key"
   $env:OPENAI_API_KEY="your-openai-api-key"
   $env:ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

## Usage

### Step 1: Prepare Your Data

Place videos in the `videos/` directory and create `data/manifest.jsonl`:

```jsonl
{"video_id": "amb_l_block", "video_path": "videos/amb l block.mp4", "goal_gt": "A", "goal_A": "pick the left block", "goal_B": "pick the right block", "scene_id": "block_scene", "task_family": "block_pick", "traj_type": "ambiguous", "notes": "Ambiguous trajectory for left block pick"}
```

**Required fields:**
- `video_id`: Unique identifier
- `video_path`: Path to video file (relative to repo root)
- `goal_gt`: Ground truth goal ("A" or "B") - used only for evaluation metrics, NOT passed to model
- `goal_A`/`goal_B`: Task-agnostic goal descriptions (e.g., "pick the left block")
- `scene_id`, `task_family`, `traj_type`, `notes`: Metadata for analysis

### Step 2: Validate Manifest

Before running evaluation, validate your manifest:

```bash
python scripts/validate_manifest.py --manifest data/manifest.jsonl
```

This checks:
- All video files exist
- All required fields present
- No duplicate video_ids
- Valid goal_gt values

### Step 3: Run Evaluation (CANONICAL PIPELINE)

**Main entrypoint:** `scripts/eval_dataset.py`

#### Single-Frame Mode (Baseline - Default)

Evaluates each timestamp independently with NO temporal context:

```bash
# Evaluate entire videos (all seconds) - single frame per timestamp
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --out outputs/results_single_frame.jsonl

# Or evaluate first k seconds only
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k 5 \
  --out outputs/results_k5.jsonl
```

#### Prefix-Frames Mode (Temporal Context)

Evaluates each timestamp with ALL prior frames (cumulative visual context):

```bash
# Evaluate with temporal context - send frames [0..t] at each timestamp t
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --mode prefix_frames \
  --out outputs/results_prefix_frames.jsonl

# Save frames to disk for verification
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k 10 \
  --mode prefix_frames \
  --save-frames \
  --out outputs/results_with_frames.jsonl
```

**Evaluation Modes:**
- `single_frame` (default): At time t, send ONLY the frame at t_sec. No temporal memory, memoryless evaluation.
- `prefix_frames`: At time t, send ALL frames from 0 to t (inclusive) in chronological order. Model observes cumulative motion.

**Frame Caching:**
- `--save-frames`: Exports sampled frames to `outputs/frames/{video_id}/t_{t_sec:03d}.png` for inspection

**Other Options:**
```bash
# Use different Gemini model
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --model gemini-2.5-pro \
  --out outputs/results_pro.jsonl

# Use OpenAI GPT-4o
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --provider openai \
  --out outputs/results_gpt4o.jsonl

# Use Anthropic Claude
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --provider anthropic \
  --out outputs/results_claude.jsonl
```

**Output artifacts:**
- `outputs/results_*.jsonl`: Per-frame evaluations with full API metadata (includes `evaluation_mode` field)
- `outputs/run_info_<uuid>.json`: Run-level provenance (git commit, timestamps, model config, evaluation_mode, save_frames flag)
- `outputs/pip_freeze.txt`: Dependency snapshot for reproducibility
- `outputs/frames/{video_id}/t_{t_sec:03d}.png`: Cached frames (if --save-frames used)

### Step 4: Generate Analysis Report (Optional)

```bash
python scripts/analyze_jsonl.py \
  outputs/results_all.jsonl \
  --output reports/analysis.md
```

Creates markdown report with:
- Statistics (choice distribution, confidence ranges)
- High-confidence moments
- Frame-by-frame table
- Trajectory interpretation

### Step 5: Compute Metrics (if you have human annotations)

```bash
python scripts/compute_iou.py \
  --vlm-jsonl outputs/results_all.jsonl \
  --human-jsonl data/human_annotations.jsonl \
  --output-csv metrics.csv
```

---

## Decision Rule Explained

From model outputs `pA` and `pB`, we compute `choice` and `confidence`:

```python
max_p = max(pA, pB)
confidence = int(round(max_p * 100))

if max_p >= 0.60:
    choice = 'A' if pA >= pB else 'B'
else:
    choice = 'C'  # Uncertain
```

**Rationale:**
- **Threshold 0.60**: Represents moderate confidence (60% vs 40%). Below this, we cannot confidently distinguish between goals.
- **Choice "C"**: Indicates uncertainty when probabilities are too close or both low.
- **Tie-breaking**: When pA == pB, choose 'A' (consistent tie-breaking rule).

This logic is applied consistently in:
1. `src/gemini_vlm_eval/client.py` (primary postprocessing)
2. `src/gemini_vlm_eval/schema.py` (Pydantic validator as fallback)

---

## Anti-Leakage Guarantee

**The model NEVER sees:**
- `goal_gt` (ground truth) - stored in output for metrics only, not sent to API
- `traj_type` (ambiguous/legible label) - used for analysis only
- Video filenames that might reveal answers

**The model ONLY sees:**
- Frame image(s) (JPEG bytes): 
  - `single_frame` mode: ONE image at time t
  - `prefix_frames` mode: ALL images from 0 to t (chronological order)
- Task-agnostic goal descriptions (e.g., "pick the left block", "pick the right block")
- Timestamp `t_sec` and `video_id` (for context/logging only)
- Mode-specific instruction:
  - `single_frame`: "You are given ONLY ONE image... Use ONLY this frame. Do NOT assume you saw earlier or later frames."
  - `prefix_frames`: "You are given MULTIPLE images showing frames from t=0 to t=T... Frames are ordered from earliest to latest; you have observed the motion up to time t=T s."

**Code verification:**
- `src/gemini_vlm_eval/prompt.py`: Prompt template uses only `goal_A`, `goal_B`, `t_sec`, `video_id`, `mode`
- `src/gemini_vlm_eval/client.py`: Never passes `goal_gt` to API; prepares single image or list based on mode
- Decision logic uses only `pA`, `pB` from model output

---

## Evaluation Modes Explained

This framework supports TWO evaluation modes to study the effect of temporal context:

### Mode 1: `single_frame` (Baseline, Default)

**What it does:** At each timestamp t, send ONLY the frame at t_sec to the model.

**Use case:** Tests whether legibility can be judged from individual snapshots without temporal context. Mimics a memoryless observer who sees isolated moments.

**Example:** At t=5s, model receives one image showing the robot at 5 seconds.

**Prompt instruction:** "Use ONLY this frame. Do NOT assume you saw earlier or later frames."

### Mode 2: `prefix_frames` (Temporal Context)

**What it does:** At each timestamp t, send ALL frames from 0 to t (inclusive) in chronological order.

**Use case:** Tests whether cumulative visual context improves legibility judgments. Mimics an observer who has watched the motion unfold from the start.

**Example:** At t=5s, model receives 6 images (frames at t=0,1,2,3,4,5) showing the complete motion sequence up to that point.

**Prompt instruction:** "Frames are ordered from earliest to latest; you have observed the motion up to time t=5s."

### Comparison Summary

| Aspect | `single_frame` | `prefix_frames` |
|--------|----------------|-----------------|
| **Images sent** | 1 frame at t | All frames 0..t |
| **Temporal context** | None (memoryless) | Cumulative (full history) |
| **API cost** | Lower (1 image/call) | Higher (N images at t=N) |
| **Use case** | Snapshot legibility | Motion-based legibility |
| **Research question** | Can single frames reveal intent? | Does observing motion help? |

### When to Use Each Mode

- **Use `single_frame`** for:
  - Baseline evaluation
  - Testing snapshot-based legibility
  - Lower-cost exploration
  - Comparing against static image understanding

- **Use `prefix_frames`** for:
  - Motion-aware legibility evaluation
  - Testing the role of temporal dynamics
  - Comparing how legibility evolves over time
  - Studying when motion becomes distinguishable

---

## Reproducibility

Every evaluation run captures:
- **Git state**: Commit hash + dirty flag
- **Dependencies**: pip freeze snapshot
- **Model config**: Model name, temperature, top_p, top_k, max_tokens, etc.
- **Timestamps**: Run start/end (UTC)
- **Platform**: Python version, OS, OpenCV version, google-genai version
- **Machine**: CPU, cores, RAM (optional)
- **Command**: Full CLI command used

All stored in `outputs/run_info_<uuid>.json`.

**To reproduce results:**
1. Check out the same git commit
2. Install from `pip_freeze.txt`: `pip install -r outputs/pip_freeze.txt`
3. Use same model and manifest
4. Run same command from `run_info.json`

**Note**: Gemini API may have slight nondeterminism even with `temperature=0.0`. We capture request/response IDs when available for traceability.

---

## Project Structure

```
gemini_vlm_eval/
├── src/gemini_vlm_eval/
│   ├── client.py          # Gemini API client + postprocessing
│   ├── prompt.py          # Task-agnostic evaluation prompt
│   ├── schema.py          # Pydantic models (ManifestEntry, EvaluationResult)
│   └── video.py           # Frame extraction utilities
├── scripts/
│   ├── eval_dataset.py    # ✓ CANONICAL: Manifest-driven batch evaluation
│   ├── validate_manifest.py  # Manifest validation
│   ├── analyze_jsonl.py   # Generate markdown reports
│   ├── compute_iou.py     # Compute IoU vs human annotations
│   ├── eval_times.py      # Evaluate specific timestamps
│   ├── eval_image.py      # Single-image evaluation (testing)
│   ├── eval_video.py      # DEPRECATED: Use eval_dataset.py
│   ├── eval_folder.py     # DEPRECATED: Use eval_dataset.py
│   ├── batch_evaluate_all.py  # DEPRECATED: Use eval_dataset.py
│   └── evaluate_video.py  # Single-video with full provenance (testing)
├── data/
│   └── manifest.jsonl     # Video metadata
├── videos/                # Video files
├── outputs/               # Evaluation results + provenance
├── reports/               # Analysis reports
│   └── pipeline_audit.md  # Full audit report (read this!)
├── pyproject.toml
└── README.md
```

---

## Advanced Usage

### Evaluate Specific Timestamps

```bash
python scripts/eval_times.py \
  --video-id amb_l_block \
  --times 1,3,5,7,9,11 \
  --out outputs/amb_l_block_targeted.jsonl
```

### Single Image Evaluation (Testing)

```bash
python scripts/eval_image.py path/to/image.jpg
```

### Custom Sampling Rate

Edit `extract_frames()` call in your script:
```python
frames = extract_frames(video_path, sample_rate_seconds=0.5)  # 2fps
```

---

Configure `VIDEO_ID` and `VIDEO_PATH` at the top of `scripts/evaluate_video.py`.

Artifacts (written to `outputs/` and `reports/`):
- `outputs/{video_id}_all.jsonl`: Per-frame results with probabilities, choice/confidence, cues, legible flag, and API metadata.
- `reports/{video_id}_analysis.md`: Markdown summary with stats, high-confidence moments, and frame table.
- `outputs/run_info.json`: Run-level provenance (run_id, timestamps, git commit, dirty flag, python/OS/platform, OpenCV, google-genai, CLI, api_key_source, pip snapshot path, machine info).
- `outputs/pip_freeze.txt`: Dependency snapshot for reproducibility.---

## Advanced Usage

### Compare Single-Frame vs Prefix-Frames

Run both modes and compare results:

```bash
# Baseline: single-frame evaluation
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --mode single_frame \
  --out outputs/results_single.jsonl

# Temporal context: prefix-frames evaluation  
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k all \
  --mode prefix_frames \
  --out outputs/results_prefix.jsonl

# Generate reports for both
python scripts/analyze_jsonl.py outputs/results_single.jsonl --output reports/analysis_single.md
python scripts/analyze_jsonl.py outputs/results_prefix.jsonl --output reports/analysis_prefix.md
```

### Save Frames for Inspection

Export the exact frames sent to the model:

```bash
python scripts/eval_dataset.py \
  --manifest data/manifest.jsonl \
  --k 10 \
  --mode prefix_frames \
  --save-frames \
  --out outputs/results_with_frames.jsonl

# Frames saved to: outputs/frames/{video_id}/t_{t_sec:03d}.png
```

### Single Video Testing

Edit configuration variables in [evaluate_video.py](scripts/evaluate_video.py):

```python
VIDEO_ID = "amb_l_block"
VIDEO_PATH = r"videos/amb l block.mp4"
MODE = "prefix_frames"  # or "single_frame"
SAVE_FRAMES = True
```

Then run:
```bash
python scripts/evaluate_video.py
```

Produces:
- `outputs/{video_id}_all.jsonl`
- `reports/{video_id}_analysis.md`
- `outputs/run_info.json`
- `outputs/pip_freeze.txt`
- `outputs/frames/{video_id}/t_*.png` (if SAVE_FRAMES=True)

### Batch Folder Evaluation (DEPRECATED)

**Note:** `eval_folder.py` is deprecated. Use `eval_dataset.py` with a manifest instead.

For legacy workflows:

```bash
python scripts/eval_folder.py videos/ --out outputs/dataset_results.jsonl
```

### Computing Metrics

Once you have human annotations in the same JSONL format:

```bash
python scripts/compute_iou.py \
  --vlm-jsonl outputs/vlm_results.jsonl \
  --human-jsonl data/human_annotations.jsonl \
  --output-csv metrics.csv
```

## Dataset Format

### Manifest File (`data/manifest.jsonl`)
Each line contains metadata for one video in JSON format:

```json
{
  "video_id": "amb_l_block",
  "video_path": "videos/amb l block.mp4",
  "goal_gt": "A",
  "goal_A": "pick the left block",
  "goal_B": "pick the right block",
  "scene_id": "block_scene",
  "task_family": "block_pick",
  "traj_type": "ambiguous",
  "notes": "Ambiguous trajectory for left block pick"
}
```

**Fields:**
- `video_id`: Unique identifier
- `video_path`: Relative path to video file
- `goal_gt`: Ground truth goal ("A" or "B")
- `goal_A`/`goal_B`: Descriptions of the two goals
- `scene_id`: Scene identifier
- `task_family`: Task type (e.g., "block_pick", "drawer_close")
- `traj_type`: Trajectory type (e.g., "ambiguous", "legible")
- `notes`: Additional information

### Creating Manifest
1. Place videos in `videos/` directory
2. Create `data/manifest.jsonl` with one JSON object per line
3. Ensure all `video_path` fields point to existing files

## Usage

### Evaluate Dataset for k Seconds

```bash
# Evaluate first 3 seconds of all videos
python scripts/eval_dataset.py --manifest data/manifest.jsonl --k 3 --out outputs/vlm_results_k3.jsonl

# Evaluate first 5 seconds
python scripts/eval_dataset.py --manifest data/manifest.jsonl --k 5 --out outputs/vlm_results_k5.jsonl

# Evaluate entire videos
python scripts/eval_dataset.py --manifest data/manifest.jsonl --k all --out outputs/vlm_results_all.jsonl
```

### Output Format

Results are stored in JSONL format with comprehensive metadata:

```json
{
  "video_id": "amb_l_block",
  "video_path": "videos/amb l block.mp4",
  "goal_gt": "A",
  "goal_A": "pick the left block",
  "goal_B": "pick the right block",
  "scene_id": "block_scene",
  "task_family": "block_pick",
  "traj_type": "ambiguous",
  "t_sec": 2,
  "frame_idx": 58,
  "pA": 0.75,
  "pB": 0.25,
  "choice": "A",
  "confidence": 75,
  "cue": "Robot arm oriented towards left block",
  "legible": "legible_now"
}
```

**Key Fields:**
- `pA`/`pB`: Model's probability estimates for each goal
- `choice`: Predicted goal ("A", "B", or "C" for uncertain)
- `confidence`: Confidence score (0-100)
- Decision rule: `choice = argmax(pA,pB)` if `max(pA,pB) >= 0.60`, else "C"

## Metrics

The `compute_iou.py` script calculates:

### IoU (Intersection over Union)
Measures agreement between VLM and human judgments of legibility over time:
- Computed per video as overlap of legible timestamp sets
- Range: 0.0 (no agreement) to 1.0 (perfect agreement)

### Time-to-Legibility
Earliest timestamp where motion becomes legible:
- VLM time-to-legibility
- Human time-to-legibility
- Useful for analyzing prediction speed

### Additional Statistics
- Count of legible timestamps per video
- Per-video breakdowns for detailed analysis

## API Reference

### Core Classes

#### `GeminiClient`
Handles communication with Gemini VLM API.

```python
from gemini_vlm_eval.client import GeminiClient

client = GeminiClient(model="gemini-1.5-flash")
result = client.evaluate_frame(image_bytes, video_path, t_sec, frame_num)
```

#### `EvaluationResult`
Pydantic model for structured results.

```python
from gemini_vlm_eval.schema import EvaluationResult

result = EvaluationResult(
    video="path/to/video.mp4",
    t_sec=2.0,
    frame=60,
    choice="A",
    confidence=85,
    cue="Visual cue description",
    legible="legible_now"
)
```

### Evaluation Functions

#### `evaluate_video()`
Process a single video file.

#### `evaluate_folder()`
Process all videos in a directory.

#### `evaluate_image()`
Evaluate a single image (for testing).

## Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key (required for `--provider google`)
- `OPENAI_API_KEY`: OpenAI API key (required for `--provider openai`)
- `ANTHROPIC_API_KEY`: Anthropic API key (required for `--provider anthropic`)

### Provider & Model Selection

The `--provider` flag selects which VLM backend to use. Default models per provider:

| Provider | Default model | Other options |
|----------|--------------|---------------|
| `google` (default) | `gemini-2.5-flash` | `gemini-2.5-pro`, `gemini-3-pro-preview`, `gemini-3.1-pro-preview`, `gemini-pro-latest` |
| `openai` | `gpt-4o` | any OpenAI vision model |
| `anthropic` | `claude-opus-4-5` | any Anthropic Claude vision model |

## Development

### Project Structure
```
gemini-vlm-eval/
├── src/gemini_vlm_eval/
│   ├── __init__.py
│   ├── client.py           # Gemini API client + postprocessing
│   ├── openai_client.py    # OpenAI GPT-4o client
│   ├── anthropic_client.py # Anthropic Claude client
│   ├── config.py           # API key loading + model defaults
│   ├── prompt.py           # Evaluation prompts
│   ├── runner.py           # Main evaluation logic
│   ├── schema.py           # Data models (Pydantic)
│   └── video.py            # Frame extraction utilities
├── scripts/
│   ├── eval_dataset.py         # ✓ CANONICAL: multi-provider batch evaluation
│   ├── multi_model_evaluation.py  # Compare multiple Gemini models
│   ├── compare_all_providers.py   # Cross-provider comparison
│   ├── validate_manifest.py
│   ├── analyze_jsonl.py
│   ├── compute_iou.py
│   ├── eval_video.py / eval_folder.py  # DEPRECATED
│   └── evaluate_video.py  # Single-video with provenance (testing)
├── outputs/               # Results directory
├── videos/                # Video files
├── data/manifest.jsonl    # Video metadata
├── pyproject.toml
└── README.md
```

### Running Tests
```bash
# Install test dependencies
pip install pytest

# Run tests
pytest
```

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

### Code Style
This project follows PEP 8 guidelines. Use `black` for code formatting:

```bash
pip install black
black src/ scripts/
```

## Research Applications

### Human-Robot Collaboration
- Evaluate robot motion clarity in shared workspaces
- Optimize trajectories for better human understanding
- Study the impact of legibility on task performance

### Motion Planning
- Use VLM feedback as a critic during reinforcement learning
- Generate legible motion primitives
- Validate legibility in simulation before real-world deployment

### Cognitive Science
- Compare human vs. VLM legibility judgments
- Study what visual features contribute to motion understanding
- Investigate cross-cultural differences in motion interpretation

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gemini_vlm_eval,
  title = {Gemini VLM Evaluation for Robot Motion Legibility},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/gemini-vlm-eval}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google DeepMind for the Gemini VLM API
- Research team members for dataset collection and annotation
- Open-source community for the underlying libraries

## Contact

For questions or collaboration opportunities:
- Email: your.email@example.com
- GitHub Issues: [Report bugs or request features](https://github.com/your-username/gemini-vlm-eval/issues)

---

**Note**: This is research software. API costs may apply when using Google Gemini, OpenAI, or Anthropic services. Ensure you have appropriate API quotas and monitor usage for whichever provider(s) you run.