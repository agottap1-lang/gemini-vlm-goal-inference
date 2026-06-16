#!/usr/bin/env python3
"""
Generate a clean, research-paper-quality VLM pipeline flow diagram.
Replaces vlm_flow_simplified.png with a publication-ready version.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "analysis_results_2"
OUTPUT_DIR.mkdir(exist_ok=True)
FRAMES_DIR = Path(__file__).parent.parent / "outputs" / "frames" / "amb_l_block"

# ── Typography ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "figure.dpi": 150,
})

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG        = "#FFFFFF"
C_FRAME_BD  = "#1E40AF"   # deep blue – frame borders & "Input Frames" label
C_FRAME_BG  = "#EFF6FF"   # very light blue – frame placeholder fill
C_VLM       = "#1E3A5F"   # dark navy – VLM box fill
C_OUT       = "#145239"   # dark forest green – output box fill
C_ARROW     = "#1F2937"   # near-black – arrows & arrow labels
C_TITLE     = "#111827"   # titles / section headers


# ── Helpers ───────────────────────────────────────────────────────────────────

def embed_frame(ax, frame_path, x, y, w, h):
    """Place a video frame using ax.imshow with data-coordinate extent."""
    try:
        img = np.array(Image.open(frame_path).convert("RGB"))
        ax.imshow(img, extent=[x, x + w, y, y + h],
                  aspect="auto", zorder=2, interpolation="bilinear")
    except Exception:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            facecolor=C_FRAME_BG,
            edgecolor=C_FRAME_BD,
            linewidth=1.5, zorder=3,
        ))
        ax.text(x + w / 2, y + h / 2, "Frame",
                ha="center", va="center", fontsize=9, color=C_FRAME_BD)

    # Crisp border on top
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        facecolor="none",
        edgecolor=C_FRAME_BD,
        linewidth=2.0, zorder=5,
    ))


def solid_box(ax, cx, cy, w, h, fill, lines, fontsizes=None):
    """
    Solid-fill rounded box with white text.
    lines     – list of strings, top to bottom
    fontsizes – list of font sizes matching lines; defaults to 13 each
    """
    if fontsizes is None:
        fontsizes = [13] * len(lines)

    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=fill,
        edgecolor=fill,
        linewidth=0, zorder=3,
    ))
    # Thin white inner border for depth
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2 + 0.02, cy - h / 2 + 0.04), w - 0.04, h - 0.08,
        boxstyle="round,pad=0.04",
        facecolor="none",
        edgecolor="white",
        linewidth=1.2, alpha=0.4, zorder=4,
    ))

    # Fixed line spacing so text is visually tight regardless of box height
    LINE_H = 0.26   # data units between consecutive text baselines
    n = len(lines)
    total_h = LINE_H * (n - 1)
    y_top = cy + total_h / 2
    for i, (text, fs) in enumerate(zip(lines, fontsizes)):
        ax.text(cx, y_top - i * LINE_H, text,
                ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white", zorder=5)


def arrow(ax, x0, x1, y, label=""):
    """Bold directional arrow with an optional bold label above."""
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_ARROW,
            lw=2.0,
            mutation_scale=22,
        ),
        zorder=4,
    )
    if label:
        mid = (x0 + x1) / 2
        ax.text(
            mid, y + 0.14, label,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=C_ARROW,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="#F9FAFB",
                edgecolor=C_ARROW,
                linewidth=0.8,
            ),
            zorder=5,
        )


# ── Main figure ───────────────────────────────────────────────────────────────

def build_figure():
    # ── Canvas ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 3.4))
    fig.patch.set_facecolor(C_BG)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)
    ax.set_xlim(0, 16)
    ax.set_ylim(0.15, 3.35)
    ax.axis("off")

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(8.0, 3.28, "VLM Legibility Evaluation Pipeline",
            ha="center", va="top",
            fontsize=17, fontweight="bold", color=C_TITLE)

    # ── Layout constants ─────────────────────────────────────────────────────
    Y_MID     = 1.65    # vertical centre of all main elements
    FW, FH    = 1.05, 1.05   # frame width / height in data units
    GAP       = 0.22    # gap between frames
    X_START   = 0.40    # left edge of first frame
    ARROW_LEN = 1.55    # length of each connecting arrow

    timepoints = [0, 6, 8, 12]
    n_frames = len(timepoints)
    total_frames_w = n_frames * FW + (n_frames - 1) * GAP

    # ── Input frames ─────────────────────────────────────────────────────────
    for i, t in enumerate(timepoints):
        fx = X_START + i * (FW + GAP)
        fy = Y_MID - FH / 2
        embed_frame(ax, FRAMES_DIR / f"t_{t:03d}.png", fx, fy, FW, FH)
        # Timestamp label below each frame
        ax.text(fx + FW / 2, fy - 0.10, f"t = {t} s",
                ha="center", va="top",
                fontsize=11, fontweight="bold", color=C_TITLE)

    # "Input Frames" section header with spanning line above all frames
    frames_cx = X_START + total_frames_w / 2
    bracket_y = Y_MID + FH / 2 + 0.08
    ax.plot([X_START, X_START + total_frames_w], [bracket_y, bracket_y],
            color=C_FRAME_BD, lw=1.5, solid_capstyle="round", zorder=3)
    ax.text(frames_cx, bracket_y + 0.08, "Input Frames",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=C_FRAME_BD)

    # ── Arrow 1: frames → VLM ────────────────────────────────────────────────
    x0_arr1 = X_START + total_frames_w + 0.18
    VLM_W, VLM_H = 2.60, 1.40
    x1_arr1 = x0_arr1 + ARROW_LEN
    VLM_CX = x1_arr1 + 0.08 + VLM_W / 2
    arrow(ax, x0_arr1, x1_arr1, Y_MID, "Prefix frames")

    # ── VLM box ───────────────────────────────────────────────────────────────
    solid_box(ax, VLM_CX, Y_MID, VLM_W, VLM_H, C_VLM,
              ["Vision-Language", "Model  (Gemini)"],
              fontsizes=[13, 13])

    # ── Arrow 2: VLM → Output ─────────────────────────────────────────────────
    x0_arr2 = VLM_CX + VLM_W / 2 + 0.18
    OUT_W, OUT_H = 2.65, 1.40
    x1_arr2 = x0_arr2 + ARROW_LEN
    OUT_CX = x1_arr2 + 0.08 + OUT_W / 2
    arrow(ax, x0_arr2, x1_arr2, Y_MID, "Goal inference")

    # ── Output box ────────────────────────────────────────────────────────────
    solid_box(ax, OUT_CX, Y_MID, OUT_W, OUT_H, C_OUT,
              ["Predicted Goal: A", "p(A) = 0.82, p(B) = 0.18"],
              fontsizes=[14, 11])

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_path = OUTPUT_DIR / "vlm_flow_simplified.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none")
    print(f"Saved: {out_path}")
