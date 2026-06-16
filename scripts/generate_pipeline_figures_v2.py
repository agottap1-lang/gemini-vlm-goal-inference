#!/usr/bin/env python3
"""
Generate two clean, presentation-ready figures:
  fig1_prompt_flow.png       — Pipeline overview (minimal text, big visuals)
  fig2_pipeline_formulas.png — Formulas front-and-centre (one per card)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "thesis_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})

C_MANIFEST = "#2563EB"   # blue
C_VIDEO    = "#7C3AED"   # purple
C_PROMPT   = "#0F766E"   # teal
C_VLM      = "#B45309"   # amber
C_OUT      = "#15803D"   # green
C_ROSE     = "#9F1239"   # rose
C_TEXT     = "#1E293B"
C_GRAY     = "#64748B"
C_BG       = "#FFFFFF"


# ─── helpers ─────────────────────────────────────────────────────────────────

def card(ax, cx, cy, w, h, color, title, bullets, title_fs=12, body_fs=10.5):
    """Card with solid colour header and white body."""
    hh = h * 0.28
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.012", linewidth=2,
        edgecolor=color, facecolor="white", zorder=3))
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy + h/2 - hh), w, hh,
        boxstyle="round,pad=0.005", linewidth=0,
        facecolor=color, zorder=4))
    ax.text(cx, cy + h/2 - hh/2, title,
            ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color="white", zorder=5)
    step = h * 0.72 / (len(bullets) + 1)
    for i, b in enumerate(bullets):
        ax.text(cx, cy + h/2 - hh - step * (i + 0.8), b,
                ha="center", va="center",
                fontsize=body_fs, color=C_TEXT, zorder=5)


def harrow(ax, x0, x1, y, label=""):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY,
                                lw=2.0, mutation_scale=16), zorder=2)
    if label:
        ax.text((x0+x1)/2, y + 0.028, label,
                ha="center", va="bottom", fontsize=9.5,
                color=C_GRAY, style="italic")


def fcell(ax, cx, cy, w, h, color, tag, lines, note=""):
    """Formula cell: tag strip on top, formula lines centred, note below box."""
    TH = h * 0.22
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.010", linewidth=2,
        edgecolor=color,
        facecolor=matplotlib.colors.to_rgba(color, 0.07), zorder=3))
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy + h/2 - TH), w, TH,
        boxstyle="round,pad=0.004", linewidth=0,
        facecolor=matplotlib.colors.to_rgba(color, 0.88), zorder=4))
    ax.text(cx, cy + h/2 - TH/2, tag,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=5)
    # distribute formula lines evenly in the body area
    body = h * (1 - 0.22)
    keep = [l for l in lines if l.strip()]
    n = len(keep)
    for i, line in enumerate(keep):
        fy = cy + h/2 - TH - body * (i + 1) / (n + 1)
        ax.text(cx, fy, line,
                ha="center", va="center",
                fontsize=12, color="#1e3a5f",
                fontfamily="monospace", zorder=5)
    # note sits BELOW the box, not inside
    if note:
        ax.text(cx, cy - h/2 - 0.014, note,
                ha="center", va="top",
                fontsize=8.5, color=C_GRAY, style="italic", zorder=5)


import matplotlib


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1
# ═════════════════════════════════════════════════════════════════════════════
def make_figure1():
    fig, ax = plt.subplots(figsize=(20, 7.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); fig.patch.set_facecolor(C_BG)

    ax.text(0.5, 0.97,
            "How the Pipeline Analyses Any Robot Video",
            ha="center", va="top",
            fontsize=17, fontweight="bold", color=C_TEXT)

    CW, CH, CY = 0.135, 0.60, 0.52
    xs = [0.09, 0.28, 0.50, 0.72, 0.91]

    card(ax, xs[0], CY, CW, CH, C_MANIFEST, "manifest.jsonl",
         ["video file path",
          "Goal A  (text label)",
          "Goal B  (text label)",
          "ground truth goal",
          "trajectory type"],
         title_fs=11, body_fs=9.5)

    card(ax, xs[1], CY, CW, CH, C_VIDEO, "Frame Extractor",
         ["load video",
          "sample 1 frame / sec",
          "t = 0, 1, 2, … seconds",
          "output: JPEG image"],
         title_fs=11, body_fs=9.5)

    card(ax, xs[2], CY, CW, CH, C_PROMPT, "Prompt Builder",
         ["insert Goal A & B",
          "insert timestamp t",
          'ask: "what is pA / pB?"',
          'ask: "legible yet?"'],
         title_fs=11, body_fs=9.5)

    card(ax, xs[3], CY, CW, CH, C_VLM, "Gemini VLM",
         ["sees prompt + image",
          "outputs pA  (0 → 1)",
          "outputs pB  (0 → 1)",
          'outputs legible? + cue'],
         title_fs=11, body_fs=9.5)

    card(ax, xs[4], CY, CW, CH, C_OUT, "Result",
         ["choice: A / B / C",
          "confidence: 0–100",
          "correct? ✓ / ✗",
          "saved to .jsonl"],
         title_fs=11, body_fs=9.5)

    gap = 0.005
    harrow(ax, xs[0]+CW/2+gap, xs[1]-CW/2-gap, CY, "path + goals")
    harrow(ax, xs[1]+CW/2+gap, xs[2]-CW/2-gap, CY, "image + t")
    harrow(ax, xs[2]+CW/2+gap, xs[3]-CW/2-gap, CY, "prompt + image")
    harrow(ax, xs[3]+CW/2+gap, xs[4]-CW/2-gap, CY, "pA, pB, legible")

    # Key insight
    ax.add_patch(FancyBboxPatch(
        (0.10, 0.055), 0.80, 0.082,
        boxstyle="round,pad=0.010", linewidth=1.5,
        edgecolor=C_MANIFEST,
        facecolor=matplotlib.colors.to_rgba(C_MANIFEST, 0.07), zorder=2))
    ax.text(0.5, 0.096,
            "To test a new video:  add one row to  manifest.jsonl  —  no code changes needed",
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=C_MANIFEST)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUTPUT_DIR / "fig1_prompt_flow.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2
# ═════════════════════════════════════════════════════════════════════════════
def make_figure2():
    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); fig.patch.set_facecolor(C_BG)

    ax.text(0.5, 0.975,
            "Pipeline Formulas",
            ha="center", va="top",
            fontsize=18, fontweight="bold", color=C_TEXT)

    def sec(y, txt, color):
        ax.text(0.5, y, txt,
                ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.28",
                          facecolor=matplotlib.colors.to_rgba(color, 0.10),
                          edgecolor=color, linewidth=1.4))

    # ── Section 1: VLM outputs ────────────────────────────────────────────────
    sec(0.930, "① What the VLM returns (per frame)", C_VLM)

    fcell(ax, 0.265, 0.825, 0.50, 0.185, C_VLM,
          "Goal probabilities",
          ["pA  =  P( Goal A  |  frames 0 … t )",
           "pB  =  P( Goal B  |  frames 0 … t )",
           "pA  +  pB  =  1"],
          note="Both probabilities must sum to 1 — enforced by the prompt")

    fcell(ax, 0.775, 0.825, 0.42, 0.185, C_VLM,
          "Legibility & cue",
          ['legible  ∈  { "legible_now",  "not_legible_yet" }',
           'cue  =  e.g.  "arm curves left"'],
          note="VLM decides if goal is readable from arm path NOW")

    # ── Section 2: hardcoded decisions ───────────────────────────────────────
    sec(0.673, "② Hardcoded pipeline decisions", C_ROSE)

    fcell(ax, 0.245, 0.567, 0.46, 0.185, C_ROSE,
          "Choice  — which goal did the VLM pick?",
          ["choice = A   if  pA ≥ pB  and  max(pA,pB) ≥ 0.52",
           "choice = B   if  pB > pA  and  max(pA,pB) ≥ 0.52",
           "choice = C   if  max(pA, pB) < 0.52"],
          note='"C" = no commitment  →  always counted as wrong')

    fcell(ax, 0.765, 0.567, 0.44, 0.185, C_ROSE,
          "Confidence  — how sure?",
          ["confidence = round( max(pA, pB) × 100 )",
           "e.g.  pA = 0.78  →  78 / 100"],
          note="0 = completely unsure   100 = fully committed")

    # ── Section 3: metrics ────────────────────────────────────────────────────
    sec(0.413, "③ How we measure performance", C_OUT)

    fcell(ax, 0.175, 0.295, 0.315, 0.200, C_OUT,
          "Correct?",
          ["correct = 1   if  choice == goal_gt",
           "correct = 0   otherwise"],
          note="goal_gt is ground truth from manifest")

    fcell(ax, 0.500, 0.295, 0.315, 0.200, C_OUT,
          "Time-to-Legibility",
          ["TTL = first t  where",
           "      legible = legible_now",
           "Lower TTL → goal readable earlier"],
          note="Compare legible vs ambiguous trajectories")

    fcell(ax, 0.825, 0.295, 0.315, 0.200, C_OUT,
          "Temporal IoU",
          ["IoU = |L_VLM ∩ L_Human|",
           "      ─────────────────",
           "      |L_VLM ∪ L_Human|"],
          note="Agreement on WHICH frames are legible")

    # ── legend strip ─────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (0.04, 0.035), 0.92, 0.095,
        boxstyle="round,pad=0.008", linewidth=1.2,
        edgecolor="#CBD5E1", facecolor="#F8FAFC", zorder=2))
    ax.text(0.5, 0.112, "Trajectory types:",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_TEXT)
    ax.text(0.27, 0.070,
            "Legible  —  arm curves toward the goal early",
            ha="center", va="center", fontsize=10.5, color=C_TEXT)
    ax.text(0.70, 0.070,
            "Ambiguous  —  arm path stays neutral / straight",
            ha="center", va="center", fontsize=10.5, color=C_TEXT)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUTPUT_DIR / "fig2_pipeline_formulas.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_figure1()
    make_figure2()
    print("Done.")
