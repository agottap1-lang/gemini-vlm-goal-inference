#!/usr/bin/env python3
"""
Generate two clean, presentation-ready figures:
  fig1_prompt_flow.png    — Pipeline overview (minimal text, big visuals)
  fig2_pipeline_formulas.png — Formulas front-and-centre (one per card)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "thesis_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
})

# Palette
C_MANIFEST = "#2563EB"
C_VIDEO    = "#7C3AED"
C_PROMPT   = "#0F766E"
C_VLM      = "#B45309"
C_OUTPUT   = "#15803D"
C_ROSE     = "#9F1239"
C_BG       = "#FFFFFF"
C_TEXT     = "#1E293B"
C_GRAY     = "#64748B"


# ─────────────────────────────────────────────────────────────────────────────
# Shared drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def card(ax, cx, cy, w, h, color, title, bullets=None, title_fs=12, body_fs=10.5):
    """Rounded card: solid colour header + white body with bullet points."""
    HEADER_FRAC = 0.30
    hh = h * HEADER_FRAC
    # body
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.015",
        linewidth=2, edgecolor=color,
        facecolor="white", zorder=3))
    # header fill
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy + h/2 - hh), w, hh,
        boxstyle="round,pad=0.006",
        linewidth=0, facecolor=color, zorder=4))
    # title
    ax.text(cx, cy + h/2 - hh/2, title,
            ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color="white", zorder=5)
    # bullets
    if bullets:
        y0 = cy + h/2 - hh - (h * (1 - HEADER_FRAC) / (len(bullets) + 1))
        step = h * (1 - HEADER_FRAC) / (len(bullets) + 1)
        for i, b in enumerate(bullets):
            ax.text(cx, y0 - i * step, b,
                    ha="center", va="center",
                    fontsize=body_fs, color=C_TEXT, zorder=5)


def harrow(ax, x0, x1, y, label="", color="#374151"):
    """Horizontal arrow with optional label above."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2.0, mutation_scale=16), zorder=2)
    if label:
        ax.text((x0 + x1)/2, y + 0.022, label,
                ha="center", va="bottom", fontsize=9,
                color=color, style="italic")


def varrow(ax, x, y0, y1, label="", color="#374151"):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=14), zorder=2)
    if label:
        ax.text(x + 0.015, (y0 + y1)/2, label,
                ha="left", va="center", fontsize=9,
                color=color, style="italic")


def formula_card(ax, cx, cy, w, h, color, tag, formula_lines, note=""):
    """Card optimised for showing a formula: big centred formula text."""
    TAG_H = h * 0.20
    # outer box
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.012",
        linewidth=2, edgecolor=color,
        facecolor=matplotlib.colors.to_rgba(color, 0.06), zorder=3))
    # tag strip
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy + h/2 - TAG_H), w, TAG_H,
        boxstyle="round,pad=0.005",
        linewidth=0, facecolor=matplotlib.colors.to_rgba(color, 0.88), zorder=4))
    ax.text(cx, cy + h/2 - TAG_H/2, tag,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=5)
    # formula lines — centred, monospace feel
    body_h = h * (1 - 0.20)
    n = len(formula_lines)
    for i, line in enumerate(formula_lines):
        fy = cy + h/2 - TAG_H - body_h * (i + 0.7) / n
        ax.text(cx, fy, line,
                ha="center", va="center",
                fontsize=12.5, color="#1e3a5f",
                fontfamily="monospace", zorder=5)
    # small note at bottom
    if note:
        ax.text(cx, cy - h/2 + 0.018, note,
                ha="center", va="bottom",
                fontsize=9, color=C_GRAY, style="italic", zorder=5)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Pipeline flow  (clean, minimal)
# ═════════════════════════════════════════════════════════════════════════════
def make_figure1():
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    ax.text(0.5, 0.96,
            "How the Pipeline Analyses Any Robot Video",
            ha="center", va="top",
            fontsize=16, fontweight="bold", color=C_TEXT)

    # ── 5 cards equally spaced ────────────────────────────────────────────────
    CW, CH, CY = 0.155, 0.62, 0.45
    xs = [0.09, 0.27, 0.50, 0.73, 0.91]

    card(ax, xs[0], CY, CW, CH, C_MANIFEST, "manifest.jsonl",
         ["video file path",
          "Goal A  (text label)",
          "Goal B  (text label)",
          "ground truth goal",
          "trajectory type"],
         title_fs=11, body_fs=9.5)

    card(ax, xs[1], CY, CW, CH, C_VIDEO, "Frame Extractor",
         ["read video file",
          "sample 1 frame / sec",
          "t = 0 s, 1 s, 2 s …",
          "→ JPEG image"],
         title_fs=11, body_fs=9.5)

    card(ax, xs[2], CY, CW, CH, C_PROMPT, "Prompt Builder",
         ["fill in Goal A & B",
          "fill in timestamp t",
          "ask for pA, pB",
          "ask for legible? + cue"],
         title_fs=11, body_fs=9.5)

    card(ax, xs[3], CY, CW, CH, C_VLM, "Gemini VLM",
         ["sees prompt + image",
          "returns pA  (0→1)",
          "returns pB  (0→1)",
          'returns "legible?" + cue'],
         title_fs=11, body_fs=9.5)

    card(ax, xs[4], CY, CW, CH, C_OUTPUT, "Result",
         ["choice  A / B / C",
          "confidence  0–100",
          "correct?  ✓ / ✗",
          "saved to .jsonl"],
         title_fs=11, body_fs=9.5)

    # ── arrows ────────────────────────────────────────────────────────────────
    gap = 0.006
    harrow(ax, xs[0] + CW/2 + gap, xs[1] - CW/2 - gap, CY, "path + goals")
    harrow(ax, xs[1] + CW/2 + gap, xs[2] - CW/2 - gap, CY, "image + t")
    harrow(ax, xs[2] + CW/2 + gap, xs[3] - CW/2 - gap, CY, "prompt + image")
    harrow(ax, xs[3] + CW/2 + gap, xs[4] - CW/2 - gap, CY, "pA, pB, legible")

    # ── key insight banner ────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (0.12, 0.05), 0.76, 0.075,
        boxstyle="round,pad=0.01",
        linewidth=1.5, edgecolor=C_MANIFEST,
        facecolor=matplotlib.colors.to_rgba(C_MANIFEST, 0.07), zorder=2))
    ax.text(0.5, 0.088,
            "To evaluate a new video:  add one row to manifest.jsonl  —  no code changes",
            ha="center", va="center",
            fontsize=11, color=C_MANIFEST, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUTPUT_DIR / "fig1_prompt_flow.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Formulas  (one formula per card, big and readable)
# ═════════════════════════════════════════════════════════════════════════════
def make_figure2():
    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    ax.text(0.5, 0.975,
            "Pipeline Formulas at a Glance",
            ha="center", va="top",
            fontsize=17, fontweight="bold", color=C_TEXT)

    # ── Section label helper ──────────────────────────────────────────────────
    def sec(y, txt, color):
        ax.text(0.5, y, txt, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=matplotlib.colors.to_rgba(color, 0.10),
                          edgecolor=color, linewidth=1.2))

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1  —  What the VLM returns  (y ~ 0.80)
    # ══════════════════════════════════════════════════════════════════════════
    sec(0.935, "① What the VLM returns", C_VLM)

    # pA / pB card (wide)
    formula_card(ax, 0.27, 0.815, 0.50, 0.19, C_VLM,
                 "Goal probabilities",
                 ["pA  =  P( Goal A  |  video frames up to time t )",
                  "pB  =  P( Goal B  |  video frames up to time t )",
                  "pA  +  pB  =  1"],
                 note="pA and pB must sum to 1 — a hard constraint in the prompt")

    # legible card
    formula_card(ax, 0.77, 0.815, 0.42, 0.19, C_VLM,
                 "Legibility label + cue",
                 ['legible  ∈  { "legible_now",  "not_legible_yet" }',
                  'cue  =  short text  e.g. "arm curves left"'],
                 note='VLM judges whether the goal is readable NOW from the arm path')

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2  —  Hardcoded post-processing  (y ~ 0.57)
    # ══════════════════════════════════════════════════════════════════════════
    sec(0.685, "② Hardcoded pipeline decisions", C_ROSE)

    formula_card(ax, 0.25, 0.565, 0.46, 0.19, C_ROSE,
                 "Choice  (which goal did the VLM pick?)",
                 ["choice  =  A   if  pA ≥ pB  and  max(pA,pB) ≥ 0.52",
                  "choice  =  B   if  pB >  pA  and  max(pA,pB) ≥ 0.52",
                  "choice  =  C   if  max(pA, pB)  <  0.52  (uncertain)"],
                 note='"C" = pipeline did not commit  →  always counted wrong')

    formula_card(ax, 0.76, 0.565, 0.44, 0.19, C_ROSE,
                 "Confidence  (how sure?)",
                 ["confidence  =  round( max(pA, pB) × 100 )",
                  "",
                  "e.g.  pA=0.78  →  confidence = 78"],
                 note="Range 0–100.  Higher = more decisive VLM call")

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3  —  Evaluation metrics  (y ~ 0.30)
    # ══════════════════════════════════════════════════════════════════════════
    sec(0.430, "③ How we measure performance", C_OUTPUT)

    formula_card(ax, 0.18, 0.310, 0.32, 0.19, C_OUTPUT,
                 "Correct prediction?",
                 ["correct  =  1   if  choice == goal_gt",
                  "correct  =  0   otherwise"],
                 note="goal_gt comes from manifest — never shown to the VLM")

    formula_card(ax, 0.50, 0.310, 0.32, 0.19, C_OUTPUT,
                 "Time-to-Legibility",
                 ["TTL  =  first t  where  legible = legible_now",
                  "Lower TTL  →  goal readable earlier"],
                 note="Compared between legible vs ambiguous trajectories")

    formula_card(ax, 0.82, 0.310, 0.32, 0.19, C_OUTPUT,
                 "Temporal IoU",
                 ["IoU  =  |L_VLM ∩ L_Human|",
                  "       ─────────────────────",
                  "        |L_VLM ∪ L_Human|"],
                 note="How well VLM and humans agree on WHICH frames are legible")

    # ── bottom legend: trajectory types ──────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (0.04, 0.045), 0.92, 0.088,
        boxstyle="round,pad=0.010",
        linewidth=1.2, edgecolor="#CBD5E1",
        facecolor="#F8FAFC", zorder=2))
    ax.text(0.5, 0.106, "Trajectory types in the dataset",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_TEXT)
    ax.text(0.28, 0.068,
            "Legible  —  arm curves toward the goal early (easy to predict)",
            ha="center", va="center", fontsize=10, color=C_TEXT)
    ax.text(0.72, 0.068,
            "Ambiguous  —  arm stays neutral / straight (harder to predict)",
            ha="center", va="center", fontsize=10, color=C_TEXT)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUTPUT_DIR / "fig2_pipeline_formulas.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_figure1()
    make_figure2()
    print("Done.")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "thesis_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────────────────────
FONT_FAMILY = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})

C_MANIFEST  = "#2563EB"   # blue
C_VIDEO     = "#7C3AED"   # purple
C_PROMPT    = "#0F766E"   # teal
C_VLM       = "#B45309"   # amber
C_OUTPUT    = "#15803D"   # green
C_POSTPROC  = "#9F1239"   # rose
C_ARROW     = "#374151"   # dark grey
C_BG        = "#F9FAFB"   # near-white
C_SECTION   = "#1E293B"   # slate


def rounded_box(ax, x, y, w, h, color, label_lines, fs=10, alpha=0.18,
                text_color="white", bold_first=True, lw=1.6):
    """Draw a rounded rectangle with centred multi-line label."""
    fancy = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.015",
        linewidth=lw, edgecolor=color,
        facecolor=matplotlib.colors.to_rgba(color, alpha),
        zorder=3,
    )
    ax.add_patch(fancy)
    # Header stripe
    header_h = h * 0.28
    header = FancyBboxPatch(
        (x - w / 2, y + h / 2 - header_h), w, header_h,
        boxstyle="round,pad=0.004",
        linewidth=0, edgecolor="none",
        facecolor=matplotlib.colors.to_rgba(color, 0.85),
        zorder=4,
    )
    ax.add_patch(header)

    if label_lines:
        # First line = title (in header)
        title = label_lines[0]
        ax.text(x, y + h / 2 - header_h / 2, title,
                ha="center", va="center", fontsize=fs + 0.5,
                fontweight="bold", color="white", zorder=5)
        # Remaining lines in body
        body = "\n".join(label_lines[1:])
        ax.text(x, y - header_h * 0.6, body,
                ha="center", va="center", fontsize=fs - 0.5,
                color=C_SECTION, zorder=5, linespacing=1.55)


def arrow(ax, x0, y0, x1, y1, label="", color=C_ARROW, lw=1.8):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=14),
        zorder=2,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.015
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=8.5, color=color, style="italic", zorder=6)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Prompt-flow / System Architecture
# ═════════════════════════════════════════════════════════════════════════════
def make_figure1():
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.96,
            "Pipeline Architecture: How Manifest + Prompt Enable Analysis of Any Video",
            ha="center", va="top", fontsize=14, fontweight="bold", color=C_SECTION)

    # ── Column x positions ───────────────────────────────────────────────────
    BW = 0.155    # box width
    BH = 0.52     # box height main row
    BY = 0.50     # box centre y

    x_manifest  = 0.09
    x_extractor = 0.27
    x_prompt    = 0.50
    x_vlm       = 0.73
    x_output    = 0.91

    # ── Manifest box ─────────────────────────────────────────────────────────
    rounded_box(ax, x_manifest, BY, BW, BH, C_MANIFEST, [
        "manifest.jsonl",
        "video_id",
        "video_path",
        "goal_A  (text)",
        "goal_B  (text)",
        "goal_gt  ← ground truth",
        "scene_id",
        "task_family",
        "traj_type",
        '  ──────────────────',
        "One row per video",
        "Add a new row →",
        "evaluate any new video",
    ], fs=9, alpha=0.14)

    # ── Frame extractor box ──────────────────────────────────────────────────
    rounded_box(ax, x_extractor, BY, BW, BH, C_VIDEO, [
        "Frame Extractor",
        "Input: video_path",
        "  ──────────────────",
        "Sample every 1 s",
        "t = 0 s, 1 s, 2 s, …",
        "  ──────────────────",
        "Output per timestep:",
        "  frame_idx  (int)",
        "  t_sec  (int)",
        "  JPEG bytes",
        "  ──────────────────",
        "Mode: single_frame",
        "or  prefix_frames",
        "(all frames 0→t)",
    ], fs=9, alpha=0.14)

    # ── Prompt constructor box ───────────────────────────────────────────────
    rounded_box(ax, x_prompt, BY, BW, BH, C_PROMPT, [
        "Prompt Constructor",
        "get_instruction_prompt(",
        "  goal_A,  goal_B,",
        "  t_sec,  video_id,",
        "  mode)",
        "  ──────────────────",
        "Injects task context:",
        "• lateral arm bias",
        "• path shape cues",
        "  ──────────────────",
        "Requests JSON output:",
        "  pA, pB, cue,",
        "  legible",
        "  ──────────────────",
        "Hard constraint:",
        "  pA + pB = 1",
    ], fs=9, alpha=0.14)

    # ── Gemini VLM box ───────────────────────────────────────────────────────
    rounded_box(ax, x_vlm, BY, BW, BH, C_VLM, [
        "Gemini VLM",
        "Input:",
        "  prompt text",
        "  + image(s)",
        "  ──────────────────",
        "temperature = 0",
        "top_p = 1.0",
        "top_k = 40",
        "max_tokens = 1024",
        "  ──────────────────",
        "Raw JSON response:",
        '  {"pA": 0.78,',
        '   "pB": 0.22,',
        '   "cue": "...",',
        '   "legible": "..."}',
    ], fs=9, alpha=0.14)

    # ── Output / EvaluationResult box ────────────────────────────────────────
    rounded_box(ax, x_output, BY, BW, BH, C_OUTPUT, [
        "EvaluationResult",
        "pA, pB  (VLM)",
        "cue  (VLM)",
        "legible  (VLM)",
        "  ──────────────────",
        "choice  ← computed",
        "confidence ← computed",
        "  ──────────────────",
        "goal_gt  ← manifest",
        "traj_type ← manifest",
        "  ──────────────────",
        "Saved to .jsonl",
        "one row per",
        "(video, t_sec)",
    ], fs=9, alpha=0.14)

    # ── Arrows ───────────────────────────────────────────────────────────────
    gap = 0.005
    arrow(ax, x_manifest + BW/2 + gap, BY,
              x_extractor - BW/2 - gap, BY,
          "video_path,\ngoal_A/B/gt")

    arrow(ax, x_extractor + BW/2 + gap, BY,
              x_prompt - BW/2 - gap, BY,
          "JPEG bytes\n+ t_sec")

    arrow(ax, x_prompt + BW/2 + gap, BY,
              x_vlm - BW/2 - gap, BY,
          "prompt text\n+ image(s)")

    arrow(ax, x_vlm + BW/2 + gap, BY,
              x_output - BW/2 - gap, BY,
          "raw JSON\n(pA, pB, cue, legible)")

    # ── Bottom annotation: "generalises to any video" ────────────────────────
    ax.annotate(
        "",
        xy=(x_manifest - BW / 2 - 0.01, 0.18),
        xytext=(x_output + BW / 2 + 0.01, 0.18),
        arrowprops=dict(arrowstyle="<->", color=C_MANIFEST,
                        lw=1.8, mutation_scale=12,
                        connectionstyle="arc3,rad=0"),
        zorder=2,
    )
    ax.text(0.5, 0.14,
            "The manifest decouples video content from evaluation logic.\n"
            "To evaluate a new video: add one row to manifest.jsonl  —  no code changes needed.",
            ha="center", va="center", fontsize=10, color=C_MANIFEST,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_MANIFEST, alpha=0.9, lw=1.4))

    # ── Mode note ─────────────────────────────────────────────────────────────
    ax.text(0.5, 0.05,
            "single_frame mode: one JPEG per call  |  "
            "prefix_frames mode: all frames from t=0 to t=T sent together",
            ha="center", va="center", fontsize=9, color="#6B7280",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#D1D5DB", alpha=0.85, lw=1))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUTPUT_DIR / "fig1_prompt_flow.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Pipeline Formulas
# ═════════════════════════════════════════════════════════════════════════════
FIG2_H   = 22.0    # virtual figure height (1 data unit = FIG2_H inches)
YLIM_BOT = 0.22    # crop bottom (content ends ~0.28, this adds a margin)


def make_figure2():
    # figsize height scaled so text density stays constant despite clipping
    fig, ax = plt.subplots(figsize=(17, FIG2_H * (1.0 - YLIM_BOT)))
    ax.set_xlim(0, 1)
    ax.set_ylim(YLIM_BOT, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    ax.text(0.5, 0.985,
            "Pipeline Formulas: From VLM Outputs to Legibility Metrics",
            ha="center", va="top", fontsize=16, fontweight="bold", color=C_SECTION)

    # ──────────────────────────────────────────────────────────────────────────
    # Key sizing constants
    #   1 data unit = FIG2_H inches  (because ylim=(0,1) and figsize height=FIG2_H)
    #   Font pt → data units: pt / (72 * FIG2_H)
    # ──────────────────────────────────────────────────────────────────────────
    TITLE_H  = 0.028                        # fixed title-strip height (data units)
    BODY_FS  = 9.7                          # body font size (points)
    # height of one body line in data units: text glyph + line gap
    LINE     = BODY_FS / (72.0 * FIG2_H) + 0.003   # ≈ 0.009 per line
    HDR_H    = 0.022                        # section-header bar height
    GAP      = 0.022                        # gap between sections

    # ─── helpers ──────────────────────────────────────────────────────────────
    def sec_header(y_centre, label, color):
        ax.add_patch(FancyBboxPatch(
            (0.01, y_centre - HDR_H / 2), 0.98, HDR_H,
            boxstyle="round,pad=0.004", linewidth=0,
            facecolor=color, alpha=0.90, zorder=3))
        ax.text(0.5, y_centre, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=4)

    def fblock(x, y_top, w, h, title, lines, color, title_fs=10.5):
        """x, y_top, w, h all in data units (axes fraction).
        y_top is the TOP edge of the box."""
        y_bot = y_top - h
        ax.add_patch(FancyBboxPatch(
            (x, y_bot), w, h,
            boxstyle="round,pad=0.009", linewidth=1.6, edgecolor=color,
            facecolor=matplotlib.colors.to_rgba(color, 0.06), zorder=3))
        ax.add_patch(FancyBboxPatch(
            (x, y_top - TITLE_H), w, TITLE_H,
            boxstyle="round,pad=0.003", linewidth=0,
            facecolor=matplotlib.colors.to_rgba(color, 0.88), zorder=4))
        ax.text(x + w / 2, y_top - TITLE_H / 2, title,
                ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color="white", zorder=5)
        # body lines — step down by LINE per non-empty line, half-step for blanks
        cur = y_top - TITLE_H - 0.006
        for line in lines:
            empty = line.strip() == ""
            if not empty:
                is_code = line.startswith("  ")
                ax.text(x + 0.010, cur, line, ha="left", va="top",
                        fontsize=BODY_FS if not is_code else BODY_FS + 0.3,
                        color="#1e3a5f" if is_code else C_SECTION,
                        fontfamily="monospace" if is_code else FONT_FAMILY,
                        zorder=5)
            cur -= (LINE * 0.5 if empty else LINE)

    def varrow(x, y0, y1, label=""):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                   lw=1.6, mutation_scale=13), zorder=6)
        if label:
            ax.text(x + 0.013, (y0 + y1) / 2, label, ha="left", va="center",
                    fontsize=8.5, color=C_ARROW, style="italic", zorder=7)

    # ══════════════════════════════════════════════════════════════════════════
    # Compute heights from content (number of effective lines × LINE + margins)
    # ══════════════════════════════════════════════════════════════════════════
    def box_h(n_lines, n_empties=0):
        """Return required box height for given content."""
        return (n_lines * LINE + n_empties * LINE * 0.5
                + TITLE_H + 0.009 + 0.008)   # 0.009 top margin, 0.008 bottom

    SA_H = box_h(5, 1)   # Goal Probs: 5 real + 1 empty
    SB_H = max(box_h(8, 2), box_h(6, 2))   # Choice (8+2) vs Confidence (6+2)
    SC_H = max(box_h(6, 2), box_h(5, 2))   # Correctness vs TrajType
    SD_H = max(box_h(12, 3), box_h(12, 3)) # TTL vs IoU

    # ── Hardcoded top positions (work down from title) ────────────────────────
    SA_HDR_Y = 0.952
    SA_TOP   = SA_HDR_Y - HDR_H / 2 - 0.008
    SA_BOT   = SA_TOP - SA_H

    SB_HDR_Y = SA_BOT - GAP - HDR_H / 2
    SB_TOP   = SB_HDR_Y - HDR_H / 2 - 0.008
    SB_BOT   = SB_TOP - SB_H

    SC_HDR_Y = SB_BOT - GAP - HDR_H / 2
    SC_TOP   = SC_HDR_Y - HDR_H / 2 - 0.008
    SC_BOT   = SC_TOP - SC_H

    SD_HDR_Y = SC_BOT - GAP - HDR_H / 2
    SD_TOP   = SD_HDR_Y - HDR_H / 2 - 0.008

    # ══════════════════════════════════════════════════════════════════════════
    # ① VLM Raw Outputs
    # ══════════════════════════════════════════════════════════════════════════
    sec_header(SA_HDR_Y, "① VLM Raw Outputs  (from Gemini — one value set per frame)", C_VLM)

    fblock(0.015, SA_TOP, 0.305, SA_H,
           "Goal Probabilities",
           ["pA = P(Goal A | frames 0..t)",
            "pB = P(Goal B | frames 0..t)",
            "",
            "  pA + pB = 1    (hard constraint)",
            "  pA, pB ∈ [0, 1]"],
           C_VLM)

    fblock(0.333, SA_TOP, 0.310, SA_H,
           "Visual Cue  (cue)",
           ["cue — natural language string",
            "describing the arm's PATH SHAPE",
            "",
            'e.g. "arm bows left of center"',
            '     "no path visible yet"'],
           C_VLM)

    fblock(0.656, SA_TOP, 0.329, SA_H,
           "Legibility Label  (legible)",
           ['legible ∈ { "legible_now",',
            '            "not_legible_yet" }',
            "",
            "VLM judges if path is clear enough",
            "for a human to infer goal NOW"],
           C_VLM)

    # ══════════════════════════════════════════════════════════════════════════
    # ② Post-Processing
    # ══════════════════════════════════════════════════════════════════════════
    sec_header(SB_HDR_Y,
               "② Post-Processing  (deterministic, hardcoded in pipeline)",
               C_POSTPROC)

    fblock(0.015, SB_TOP, 0.465, SB_H,
           "Choice  (predicted goal label)",
           ["Let  max_p = max(pA, pB)",
            "",
            "  choice = A   if  max_p ≥ 0.52  and  pA ≥ pB",
            "           B   if  max_p ≥ 0.52  and  pB > pA",
            "           C   if  max_p < 0.52  (uncertain)",
            "",
            "Threshold 0.52 keeps a tiny 'truly 50/50' band",
            "Choice 'C' is always counted as incorrect"],
           C_POSTPROC)

    fblock(0.498, SB_TOP, 0.487, SB_H,
           "Confidence  (integer, 0–100)",
           ["",
            "  confidence = round( max(pA, pB) × 100 )",
            "",
            "  pA=0.78, pB=0.22  →  confidence = 78",
            "  pA=0.50, pB=0.50  →  confidence = 50",
            "  pA=0.95, pB=0.05  →  confidence = 95"],
           C_POSTPROC)

    # ══════════════════════════════════════════════════════════════════════════
    # ③ Ground Truth
    # ══════════════════════════════════════════════════════════════════════════
    sec_header(SC_HDR_Y,
               "③ Ground Truth  (from manifest.jsonl — never seen by VLM)",
               C_MANIFEST)

    fblock(0.015, SC_TOP, 0.465, SC_H,
           "Correctness  (per frame)",
           ["goal_gt ∈ { 'A', 'B' }  — true goal from manifest",
            "",
            "  correct = 1   if  choice == goal_gt",
            "            0   otherwise  (choice='C' always = 0)",
            "",
            "Accuracy = mean(correct) over frames/videos"],
           C_MANIFEST)

    fblock(0.498, SC_TOP, 0.487, SC_H,
           "Trajectory Type  (per video)",
           ["traj_type ∈ { 'legible', 'ambiguous' }",
            "",
            "Legible   — path curved toward goal from t=0;",
            "            strong lateral bias",
            "Ambiguous — path neutral/straight; late commit"],
           C_MANIFEST)

    # ══════════════════════════════════════════════════════════════════════════
    # ④ Evaluation Metrics
    # ══════════════════════════════════════════════════════════════════════════
    sec_header(SD_HDR_Y,
               "④ Evaluation Metrics  (comparing VLM legibility calls over time)",
               C_OUTPUT)

    fblock(0.015, SD_TOP, 0.465, SD_H,
           "Time-to-Legibility  (TTL)",
           ["For a video sampled at t = 0 s, 1 s, 2 s … T s",
            "",
            "  TTL_VLM   = min{ t : legible(t) = 'legible_now' }",
            "  TTL_Human = min{ t : human says legible at t }",
            "",
            "  If never legible  →  TTL = ∞  (None)",
            "",
            "  Δ TTL = TTL_VLM − TTL_Human  (alignment error)",
            "",
            "Lower TTL ⟹ goal readable earlier.",
            "Legible traj. expected to have lower TTL",
            "than ambiguous traj."],
           C_OUTPUT)

    fblock(0.498, SD_TOP, 0.487, SD_H,
           "Temporal IoU  (VLM vs Human)",
           ["Let  L_VLM   = { t : VLM says 'legible_now' }",
            "     L_Human = { t : human annotates legible }",
            "",
            "  IoU =    |L_VLM ∩ L_Human|",
            "         ─────────────────────",
            "           |L_VLM ∪ L_Human|",
            "",
            "IoU ∈ [0, 1]",
            "  1.0  →  perfect temporal agreement",
            "  0.0  →  no overlap at all",
            "",
            "Measures agreement on WHICH frames are",
            "legible, not just the first."],
           C_OUTPUT)

    # ── connecting arrows ─────────────────────────────────────────────────────
    varrow(0.17, SA_BOT - 0.001, SA_BOT - GAP * 0.5, "pA, pB →")
    varrow(0.82, SA_BOT - 0.001, SA_BOT - GAP * 0.5, "legible →")
    varrow(0.23, SB_BOT - 0.001, SB_BOT - GAP * 0.5, "choice →")
    varrow(0.50, SC_BOT - 0.001, SC_BOT - GAP * 0.5, "goal_gt →")

    fig.tight_layout(rect=[0, 0, 1, 1])
    out = OUTPUT_DIR / "fig2_pipeline_formulas.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_figure1()
    make_figure2()
    print("Done.")
