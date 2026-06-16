#!/usr/bin/env python3
"""
Generate presenter-friendly pipeline figures grounded in the current codebase.

Outputs:
  - figureA_prompt_template.(png|pdf)
  - figureB_manifest_prompt_flow.(png|pdf)
  - figureC_pipeline_formulas.(png|pdf)
  - PRESENTER_NOTES.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "analysis_results_2" / "presenter_pipeline_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))
from gemini_vlm_eval.prompt import get_instruction_prompt  # noqa: E402


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 18,
    }
)

COLOR_BG = "#F7F7F5"
COLOR_TEXT = "#1F2937"
COLOR_MUTED = "#6B7280"
COLOR_MANIFEST = "#1D4ED8"
COLOR_PROMPT = "#0F766E"
COLOR_VIDEO = "#7C3AED"
COLOR_VLM = "#C2410C"
COLOR_CODE = "#BE185D"
COLOR_ANALYSIS = "#15803D"
COLOR_LINE = "#374151"
COLOR_PANEL = "#FFFFFF"
COLOR_WARN = "#B45309"


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    title: str,
    body_lines: Iterable[str],
    body_size: float = 10.5,
    title_size: float = 12.5,
    face_alpha: float = 0.08,
) -> None:
    """Draw a rounded box with a colored title band."""

    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.7,
        edgecolor=color,
        facecolor=matplotlib.colors.to_rgba(color, face_alpha),
        zorder=2,
    )
    ax.add_patch(panel)

    header_h = min(0.12, h * 0.24)
    header = FancyBboxPatch(
        (x, y + h - header_h),
        w,
        header_h,
        boxstyle="round,pad=0.006,rounding_size=0.02",
        linewidth=0,
        edgecolor="none",
        facecolor=matplotlib.colors.to_rgba(color, 0.93),
        zorder=3,
    )
    ax.add_patch(header)

    ax.text(
        x + w / 2,
        y + h - header_h / 2,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="white",
        zorder=4,
    )

    body = "\n".join(body_lines)
    ax.text(
        x + 0.015,
        y + h - header_h - 0.018,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        color=COLOR_TEXT,
        linespacing=1.45,
        zorder=4,
    )


def arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str = "",
    color: str = COLOR_LINE,
    fontsize: float = 9.5,
) -> None:
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color=color,
        zorder=5,
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2 + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            style="italic",
            zorder=6,
        )


def save(fig: plt.Figure, stem: str) -> None:
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def load_manifest_example() -> dict:
    manifest_path = REPO_ROOT / "data" / "manifest.jsonl"
    with manifest_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    return json.loads(first_line)


def generate_prompt_template_figure() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Figure A. Prompt Template in prompt.py",
        ha="center",
        va="top",
        fontsize=21,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    ax.text(
        0.5,
        0.928,
        "What the VLM is asked to do, what comes from the manifest, and what stays hidden.",
        ha="center",
        va="top",
        fontsize=11.5,
        color=COLOR_MUTED,
    )

    rounded_box(
        ax,
        0.05,
        0.54,
        0.58,
        0.30,
        COLOR_PROMPT,
        "1. Mode-Specific Context",
        [
            'prefix_frames: "You are given MULTIPLE images showing frames from t=0 to t=T."',
            'single_frame:  "You are given ONLY ONE image captured at time t=T."',
            "",
            "Runtime inputs inserted here:",
            "  - video_id",
            "  - t_sec",
            "  - mode",
        ],
    )

    rounded_box(
        ax,
        0.05,
        0.25,
        0.58,
        0.24,
        COLOR_MANIFEST,
        "2. Goal Options Injected from manifest.jsonl",
        [
            "There are exactly two candidate goals:",
            "  Goal A: <goal_A>",
            "  Goal B: <goal_B>",
            "",
            "This is the main reason the same prompt template can be reused",
            "for every video row in the benchmark.",
        ],
    )

    rounded_box(
        ax,
        0.67,
        0.54,
        0.28,
        0.30,
        COLOR_VIDEO,
        "3. Fixed Interpretation Rule",
        [
            "The prompt tells the model to inspect the arm path shape,",
            "especially lateral bias / curvature, then estimate:",
            "  - pA",
            "  - pB",
            "  - cue",
            "  - legible",
        ],
    )

    rounded_box(
        ax,
        0.67,
        0.25,
        0.28,
        0.24,
        COLOR_VLM,
        "4. Required JSON Output",
        [
            '{"pA": ..., "pB": ..., "cue": "...",',
            ' "legible": "legible_now" | "not_legible_yet"}',
            "",
            "The prompt also asks for pA + pB = 1.",
            "Only these four fields come directly from the VLM.",
        ],
        body_size=10.2,
    )

    rounded_box(
        ax,
        0.05,
        0.06,
        0.90,
        0.12,
        COLOR_CODE,
        "Speaker Notes",
        [
            "The prompt does NOT include goal_gt or traj_type.",
            "Those fields stay in the manifest and are only used after inference for evaluation and grouping.",
            "Important nuance: the current prompt template is reusable for new two-goal videos in this benchmark,",
            "but it is not fully universal for arbitrary task families because the interpretation logic is still hand-written.",
        ],
        body_size=10.0,
    )

    save(fig, "figureA_prompt_template")


def generate_manifest_flow_figure() -> None:
    example = load_manifest_example()

    fig, ax = plt.subplots(figsize=(17, 10))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Figure B. How Manifest + Prompt Drive the Pipeline",
        ha="center",
        va="top",
        fontsize=21,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    ax.text(
        0.5,
        0.928,
        "Field provenance: what comes from the manifest, what comes from the VLM, and what is hardcoded in code.",
        ha="center",
        va="top",
        fontsize=11.5,
        color=COLOR_MUTED,
    )

    y = 0.42
    w = 0.17
    h = 0.34

    rounded_box(
        ax,
        0.03,
        y,
        w,
        h,
        COLOR_MANIFEST,
        "Manifest Row",
        [
            f'video_id: "{example["video_id"]}"',
            f'video_path: "{example["video_path"]}"',
            f'goal_gt: "{example["goal_gt"]}"',
            f'goal_A: "{example["goal_A"]}"',
            f'goal_B: "{example["goal_B"]}"',
            f'traj_type: "{example["traj_type"]}"',
            "",
            "Reusable contract:",
            "one row defines one new video.",
        ],
        body_size=10.0,
    )

    rounded_box(
        ax,
        0.23,
        y,
        w,
        h,
        COLOR_VIDEO,
        "Frame Sampling",
        [
            "extract_frames(video_path, 1.0s)",
            "",
            "target_times = 0, 1, 2, ...",
            "frame_idx = int(t * fps)",
            "",
            "single_frame  -> send frame[t]",
            "prefix_frames -> send frames[0..t]",
        ],
        body_size=10.0,
    )

    rounded_box(
        ax,
        0.43,
        y,
        w,
        h,
        COLOR_PROMPT,
        "Prompt Builder",
        [
            "get_instruction_prompt(",
            "  goal_A, goal_B, t_sec, video_id, mode",
            ")",
            "",
            "Prompt sees:",
            "  - goal_A, goal_B",
            "  - video_id, t_sec, mode",
            "",
            "Prompt does NOT see:",
            "  - goal_gt",
            "  - traj_type",
        ],
        body_size=9.8,
    )

    rounded_box(
        ax,
        0.63,
        y,
        w,
        h,
        COLOR_VLM,
        "Gemini VLM",
        [
            "Inputs:",
            "  - prompt text",
            "  - image(s)",
            "",
            "Returns JSON:",
            "  - pA",
            "  - pB",
            "  - cue",
            "  - legible",
        ],
    )

    rounded_box(
        ax,
        0.83,
        y,
        0.14,
        h,
        COLOR_CODE,
        "Code Post-Processing",
        [
            "Hardcoded from pA / pB:",
            "  max_p = max(pA, pB)",
            "  confidence = round(max_p * 100)",
            "  choice = A / B / C",
            "    using threshold 0.52",
            "",
            "legible is NOT computed here.",
        ],
        body_size=9.5,
    )

    arrow(ax, 0.20, y + h / 2, 0.23, y + h / 2, "video_path")
    arrow(ax, 0.40, y + h / 2, 0.43, y + h / 2, "frames + t_sec")
    arrow(ax, 0.60, y + h / 2, 0.63, y + h / 2, "prompt + image(s)")
    arrow(ax, 0.80, y + h / 2, 0.83, y + h / 2, "pA, pB, cue, legible")

    rounded_box(
        ax,
        0.14,
        0.08,
        0.72,
        0.20,
        COLOR_ANALYSIS,
        "Saved Result Row + Downstream Analysis",
        [
            "Each JSONL row merges three sources:",
            "  1. Manifest metadata: video_id, goal_gt, goal_A, goal_B, traj_type, scene_id, task_family",
            "  2. VLM outputs: pA, pB, cue, legible",
            "  3. Hardcoded code outputs: choice, confidence",
            "",
            "Then analysis uses goal_gt for correctness and traj_type for grouping, plus legible over time for TTL / IoU.",
        ],
        body_size=10.2,
    )

    arrow(ax, 0.10, y, 0.24, 0.28, "manifest metadata")
    arrow(ax, 0.70, y, 0.50, 0.28, "VLM outputs")
    arrow(ax, 0.90, y, 0.62, 0.28, "choice + confidence")

    ax.text(
        0.5,
        0.02,
        "Presenter framing: the manifest makes the pipeline scalable, the prompt defines the task, the VLM supplies probabilities/cues, and the code turns those probabilities into decisions.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=COLOR_MUTED,
    )

    save(fig, "figureB_manifest_prompt_flow")


def generate_formula_figure() -> None:
    fig, ax = plt.subplots(figsize=(17, 11))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Figure C. Formulas and Decision Rules Used in the Pipeline",
        ha="center",
        va="top",
        fontsize=21,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    ax.text(
        0.5,
        0.928,
        "This figure separates prompt constraints, VLM outputs, hardcoded code rules, and downstream metrics.",
        ha="center",
        va="top",
        fontsize=11.5,
        color=COLOR_MUTED,
    )

    rounded_box(
        ax,
        0.03,
        0.67,
        0.30,
        0.21,
        COLOR_PROMPT,
        "Prompt / Sampling Rules",
        [
            "Frame sampling:",
            "  target_times = 0, 1, 2, ...",
            "  frame_idx = int(t * fps)",
            "",
            "Evaluation mode:",
            "  single_frame  -> image[t]",
            "  prefix_frames -> images[0..t]",
            "",
            "Prompt asks the VLM to satisfy:",
            "  0 <= pA, pB <= 1",
            "  pA + pB = 1",
        ],
        body_size=10.0,
    )

    rounded_box(
        ax,
        0.35,
        0.67,
        0.30,
        0.21,
        COLOR_VLM,
        "Direct VLM Outputs",
        [
            "pA = P(Goal A | observed frame(s))",
            "pB = P(Goal B | observed frame(s))",
            "",
            'cue = short natural-language visual explanation',
            'legible in {"legible_now", "not_legible_yet"}',
            "",
            "Important:",
            "  legible is returned by the model,",
            "  not computed from pA / pB in code.",
        ],
        body_size=10.0,
    )

    rounded_box(
        ax,
        0.67,
        0.67,
        0.30,
        0.21,
        COLOR_CODE,
        "Hardcoded Post-Processing",
        [
            "max_p = max(pA, pB)",
            "confidence = int(round(max_p * 100))",
            "",
            "choice = 'A' if max_p >= 0.52 and pA >= pB",
            "choice = 'B' if max_p >= 0.52 and pB >  pA",
            "choice = 'C' if max_p <  0.52",
            "",
            "Tie rule:",
            "  if pA == pB and threshold is passed, choose A.",
        ],
        body_size=9.8,
    )

    rounded_box(
        ax,
        0.03,
        0.37,
        0.46,
        0.22,
        COLOR_MANIFEST,
        "Manifest-Based Evaluation Fields",
        [
            "goal_gt in {'A', 'B'}",
            "traj_type in {'legible', 'ambiguous'}",
            "",
            "Per-row correctness:",
            "  correct = 1 if choice == goal_gt else 0",
            "",
            "Because goal_gt is only A or B,",
            "choice == 'C' is always counted as incorrect.",
        ],
        body_size=10.0,
    )

    rounded_box(
        ax,
        0.51,
        0.37,
        0.46,
        0.22,
        COLOR_ANALYSIS,
        "Temporal Legibility Metrics",
        [
            "L_vlm   = {t : legible(t) == 'legible_now'}",
            "L_human = {t : human says legible at t}",
            "",
            "TTL_vlm   = min(L_vlm)   if L_vlm is not empty else None",
            "TTL_human = min(L_human) if L_human is not empty else None",
            "",
            "IoU = |L_vlm intersect L_human| / |L_vlm union L_human|",
        ],
        body_size=9.8,
    )

    rounded_box(
        ax,
        0.03,
        0.10,
        0.94,
        0.18,
        COLOR_WARN,
        "What to Say Out Loud",
        [
            "1. pA, pB, cue, and legible come from the VLM.",
            "2. choice and confidence are deterministic code outputs computed from pA and pB.",
            "3. goal_gt and traj_type come from the manifest and are used only after inference.",
            "4. The current code uses threshold 0.52 for choice, even though some docs still mention 0.60.",
        ],
        body_size=10.0,
    )

    save(fig, "figureC_pipeline_formulas")


def write_presenter_notes() -> None:
    prompt_example = get_instruction_prompt(
        "<goal_A>",
        "<goal_B>",
        5,
        "<video_id>",
        mode="prefix_frames",
    )

    note_lines: List[str] = [
        "# Presenter Notes: Manifest, Prompt, and Formulas",
        "",
        "## What the manifest does",
        "- `data/manifest.jsonl` is the dataset contract: one JSON row defines one video.",
        "- It stores `video_path`, `goal_A`, `goal_B`, `goal_gt`, `traj_type`, and metadata like `scene_id` and `task_family`.",
        "- `goal_gt` and `traj_type` are not sent to the VLM. They are used later for correctness and grouped analysis.",
        "",
        "## What the prompt does",
        "- `src/gemini_vlm_eval/prompt.py` builds the text prompt from `goal_A`, `goal_B`, `t_sec`, `video_id`, and `mode`.",
        "- The model is asked to return only four things: `pA`, `pB`, `cue`, and `legible`.",
        "- In `prefix_frames` mode, the model sees all sampled frames from `t=0` to the current `t`.",
        "- In `single_frame` mode, the model sees only one frame at the current timestamp.",
        "",
        "## Exact value provenance",
        "- From the VLM: `pA`, `pB`, `cue`, `legible`.",
        "- From hardcoded code: `choice`, `confidence`.",
        "- From the manifest: `goal_gt`, `traj_type`, `goal_A`, `goal_B`, `video_id`, `video_path`.",
        "",
        "## Exact formulas used in code",
        "- `max_p = max(pA, pB)`",
        "- `confidence = int(round(max_p * 100))`",
        "- `choice = 'A'` if `max_p >= 0.52` and `pA >= pB`",
        "- `choice = 'B'` if `max_p >= 0.52` and `pB > pA`",
        "- `choice = 'C'` if `max_p < 0.52`",
        "- `correct = 1 if choice == goal_gt else 0`",
        "- `L_vlm = {t : legible(t) == 'legible_now'}`",
        "- `TTL_vlm = min(L_vlm)` if that set is non-empty, else `None`",
        "- `IoU = |L_vlm intersect L_human| / |L_vlm union L_human|`",
        "",
        "## Important clarifications for your presentation",
        "- `legible` is not computed from `pA` and `pB` in the code. It is directly predicted by the VLM.",
        "- `choice` can be `C` even though the prompt encourages decisive probabilities, because the code keeps a small uncertainty band below `0.52`.",
        "- The current prompt template is reusable for new two-goal videos in the same benchmark setup, but it is not fully universal for arbitrary tasks because the interpretation logic is still hand-written.",
        "- Some documentation files still say the threshold is `0.60`, but the live code in `client.py` and `schema.py` uses `0.52`.",
        "",
        "## Sample prompt excerpt",
        "```text",
        *prompt_example.strip().splitlines()[:18],
        "```",
        "",
    ]

    notes_path = OUTPUT_DIR / "PRESENTER_NOTES.md"
    notes_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    print(f"Saved: {notes_path}")


def main() -> None:
    generate_prompt_template_figure()
    generate_manifest_flow_figure()
    generate_formula_figure()
    write_presenter_notes()
    print("Done.")


if __name__ == "__main__":
    main()
