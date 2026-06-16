#!/usr/bin/env python3
"""
Multi-provider comparison: Gemini vs OpenAI vs Anthropic Claude.

Usage:
  python scripts/compare_all_providers.py \
      --gemini  analysis_3/results_gemini_3_pro_preview.jsonl \
      --openai  outputs/results_gpt4o.jsonl \
      --claude  outputs/results_claude3_opus.jsonl \
      --out     outputs/provider_comparison

Reads JSONL result files (same format as eval_dataset.py output), computes
per-video and overall accuracy / IoU / flip metrics, and writes:
  - provider_comparison/comparison_table.csv
  - provider_comparison/figure_accuracy_bar.png / .pdf
  - provider_comparison/figure_per_video_heatmap.png / .pdf
  - provider_comparison/figure_radar.png / .pdf
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ground truth (same as the rest of the pipeline)
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "amb_d_drawer_close": "A",
    "amb_l_block":        "A",   # goal: move to left block
    "amb_r_block":        "B",   # goal: move to right block
    "amb_to_drawer_close":"B",
    "le_d_drawer_close":  "A",
    "le_l_block":         "A",
    "le_r_block":         "B",
    "le_t_drawer_close":  "B",
}

TRAJ_TYPE = {
    "amb_d_drawer_close": "ambiguous",
    "amb_l_block":        "ambiguous",
    "amb_r_block":        "ambiguous",
    "amb_to_drawer_close":"ambiguous",
    "le_d_drawer_close":  "legible",
    "le_l_block":         "legible",
    "le_r_block":         "legible",
    "le_t_drawer_close":  "legible",
}

HUMAN_ACCURACY = 98.25   # from prior study
HUMAN_LABEL    = "Human\n(98.25%)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    skipped = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"  ⚠️  Skipped {skipped} malformed line(s) in {path}")
    df = pd.DataFrame(rows)
    return df


def compute_metrics(df: pd.DataFrame) -> Dict:
    """Overall + per-video accuracy, IoU, flip count, no-flip rate.

    Accuracy is computed over *committed* predictions only (choice A or B).
    Choice C ("not legible yet") is excluded from accuracy numerator and
    denominator — consistent with the original study methodology.
    """
    df = df.copy()

    # Separate committed (A/B) from uncertain (C) predictions.
    committed = df[df["choice"].isin(["A", "B"])].copy()
    committed["correct"] = committed["choice"] == committed["goal_gt"]

    overall_acc = committed["correct"].mean() * 100 if len(committed) > 0 else 0.0
    n = len(committed)
    n_abstain = len(df) - n

    # per-video (over committed predictions only)
    per_video = {}
    for vid, grp in df.groupby("video_id"):
        grp_committed = grp[grp["choice"].isin(["A", "B"])].copy()
        grp_committed["correct"] = grp_committed["choice"] == grp_committed["goal_gt"]
        acc = grp_committed["correct"].mean() * 100 if len(grp_committed) > 0 else 0.0
        gt  = GROUND_TRUTH.get(vid, "?")

        # flip count: consecutive changes among committed (A/B) choices only.
        # C ("not legible yet") predictions are excluded — consistent with the
        # original study methodology where C=abstain does not count as a flip.
        committed_sorted = grp_committed.sort_values("t_sec")["choice"].tolist()
        flips   = sum(1 for a, b in zip(committed_sorted, committed_sorted[1:]) if a != b)

        # early correct: first *committed* prediction matches goal_gt from data
        if len(grp_committed) > 0:
            first_row = grp_committed.sort_values("t_sec").iloc[0]
            early_ok  = first_row["choice"] == first_row["goal_gt"]
        else:
            early_ok = False

        per_video[vid] = {
            "accuracy":      acc,
            "flips":         flips,
            "early_correct": early_ok,
            "n":             len(grp_committed),
            "n_abstain":     len(grp) - len(grp_committed),
        }

    flip_total    = sum(v["flips"]        for v in per_video.values())
    early_correct = sum(1 for v in per_video.values() if v["early_correct"])
    no_flip_rate  = sum(1 for v in per_video.values() if v["flips"] == 0) / len(per_video) * 100

    return {
        "overall_accuracy": overall_acc,
        "iou":              overall_acc,   # IoU == accuracy for binary tasks here
        "total_flips":      flip_total,
        "no_flip_rate":     no_flip_rate,
        "early_correct_pct": early_correct / len(per_video) * 100,
        "n_predictions":    n,
        "n_abstain":        n_abstain,
        "per_video":        per_video,
    }


def save_figure(fig, path_stem: Path) -> None:
    for ext in (".png", ".pdf"):
        fig.savefig(str(path_stem) + ext, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Overall accuracy bar chart with human baseline
# ---------------------------------------------------------------------------

def fig_accuracy_bar(providers: Dict[str, Dict], out_stem: Path) -> None:
    names  = list(providers.keys())
    accs   = [providers[n]["overall_accuracy"] for n in names]
    colors = ["#4285F4", "#EA4335", "#FBBC05"]   # Google blue, OpenAI red, Anthropic yellow

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=colors[:len(names)], width=0.5, zorder=3)

    # Human baseline
    ax.axhline(HUMAN_ACCURACY, color="#34A853", linestyle="--", linewidth=2,
               label=f"Human ({HUMAN_ACCURACY:.1f}%)", zorder=4)

    # Value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("VLM Legibility Accuracy by Provider\n(prefix_frames mode, best model per provider)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, out_stem / "figure_accuracy_bar")


# ---------------------------------------------------------------------------
# Figure 2: Per-video accuracy heatmap
# ---------------------------------------------------------------------------

def fig_per_video_heatmap(providers: Dict[str, Dict], out_stem: Path) -> None:
    provider_names = list(providers.keys())
    videos         = sorted(GROUND_TRUTH.keys())

    matrix = np.zeros((len(provider_names), len(videos)))
    for pi, name in enumerate(provider_names):
        per_vid = providers[name]["per_video"]
        for vi, vid in enumerate(videos):
            matrix[pi, vi] = per_vid.get(vid, {}).get("accuracy", 0.0)

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(range(len(videos)))
    ax.set_xticklabels(videos, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(provider_names)))
    ax.set_yticklabels(provider_names, fontsize=11)

    # Cell annotations
    for pi in range(len(provider_names)):
        for vi in range(len(videos)):
            val = matrix[pi, vi]
            color = "white" if val < 40 else "black"
            ax.text(vi, pi, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_title("Per-Video Accuracy by Provider", fontsize=13, fontweight="bold")

    # Traj type annotation
    for vi, vid in enumerate(videos):
        ttype = TRAJ_TYPE.get(vid, "")
        ax.text(vi, len(provider_names) - 0.5 + 0.7,
                "A" if ttype == "ambiguous" else "L",
                ha="center", va="bottom", fontsize=7, color="grey")

    fig.tight_layout()
    save_figure(fig, out_stem / "figure_per_video_heatmap")


# ---------------------------------------------------------------------------
# Figure 3: Radar / spider chart of 4 metrics
# ---------------------------------------------------------------------------

def fig_radar(providers: Dict[str, Dict], out_stem: Path) -> None:
    metrics      = ["Accuracy", "IoU", "No-Flip\nRate", "Early\nCorrect"]
    metric_keys  = ["overall_accuracy", "iou", "no_flip_rate", "early_correct_pct"]
    provider_names = list(providers.keys())

    N      = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    colors = ["#4285F4", "#EA4335", "#FBBC05"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, name in enumerate(provider_names):
        values = [providers[name][k] for k in metric_keys]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=colors[i], label=name)
        ax.fill(angles, values, alpha=0.12, color=colors[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="grey")
    ax.set_title("Provider Comparison – Key Metrics", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    fig.tight_layout()
    save_figure(fig, out_stem / "figure_radar")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare VLM providers")
    parser.add_argument("--gemini",  help="Path to Gemini JSONL results")
    parser.add_argument("--openai",  help="Path to OpenAI JSONL results")
    parser.add_argument("--claude",  help="Path to Anthropic JSONL results")
    parser.add_argument("--out",     default="outputs/provider_comparison",
                        help="Output directory (default: outputs/provider_comparison)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    providers: Dict[str, Dict] = {}
    raw_dfs: Dict[str, pd.DataFrame] = {}

    for label, path in [("Gemini 3 Pro", args.gemini),
                        ("GPT-4o",       args.openai),
                        ("Claude Opus",  args.claude)]:
        if path and Path(path).exists():
            print(f"Loading {label} from {path} ...")
            df = load_jsonl(path)
            raw_dfs[label] = df
            print(f"  → {len(df)} predictions loaded")
        else:
            if path:
                print(f"  ⚠️  {label}: file not found — {path}")
            else:
                print(f"  ⚠️  {label}: no path given, skipping")

    if not raw_dfs:
        print("No valid result files found. Exiting.")
        sys.exit(1)

    # Align to the intersection of (video_id, t_sec) across all providers
    # so comparison is fair regardless of different --k settings used per run.
    shared_keys = None
    for df in raw_dfs.values():
        keys = set(zip(df["video_id"], df["t_sec"].astype(int)))
        shared_keys = keys if shared_keys is None else shared_keys & keys

    print(f"\nAligned to {len(shared_keys)} shared (video_id, t_sec) timepoints across all providers.")

    for label, df in raw_dfs.items():
        df_aligned = df[
            df.apply(lambda r: (r["video_id"], int(r["t_sec"])) in shared_keys, axis=1)
        ].copy()
        providers[label] = compute_metrics(df_aligned)
        print(f"  {label}: {len(df_aligned)} total → {providers[label]['n_predictions']} committed, "
              f"{providers[label]['n_abstain']} abstain (C), "
              f"accuracy={providers[label]['overall_accuracy']:.2f}%")

    # ---- Save CSV ----
    rows = []
    for name, m in providers.items():
        rows.append({
            "Provider":             name,
            "Accuracy (%)":         round(m["overall_accuracy"], 2),
            "IoU (%)":              round(m["iou"], 2),
            "No-Flip Rate (%)":     round(m["no_flip_rate"], 2),
            "Early Correct (%)":    round(m["early_correct_pct"], 2),
            "Total Flips":          m["total_flips"],
            "Committed Predictions": m["n_predictions"],
            "Abstain (C) Count":    m["n_abstain"],
        })
    df_summary = pd.DataFrame(rows)
    csv_path = out_dir / "comparison_table.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table → {csv_path}")
    print(df_summary.to_string(index=False))

    # ---- Figures ----
    if len(providers) >= 1:
        fig_accuracy_bar(providers, out_dir)
        print(f"Saved accuracy bar chart → {out_dir}/figure_accuracy_bar.*")

    if len(providers) >= 1:
        fig_per_video_heatmap(providers, out_dir)
        print(f"Saved per-video heatmap  → {out_dir}/figure_per_video_heatmap.*")

    if len(providers) >= 2:
        fig_radar(providers, out_dir)
        print(f"Saved radar chart        → {out_dir}/figure_radar.*")

    print("\nDone.")


if __name__ == "__main__":
    main()
