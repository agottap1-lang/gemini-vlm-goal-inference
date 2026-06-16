#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM-as-Proxy-for-Human-Judgment Analysis
=========================================
Following the colleague's advice:
  - n = 8 trajectories (not participants)
  - Compute mean human accuracy ± SD per trajectory (across 8 participants)
  - Compute VLM accuracy per trajectory (at human study timepoints only)
  - Spearman r between mean human score and VLM score
  - Critical r for n=8 is ≥ 0.738 (two-tailed α=0.05)

Additionally:
  - Also test using VLM's continuous pA score (probability assigned to correct goal)
  - Produce 4 publication-quality figures saved to analysis_results_2/vlm_proxy/
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
from pathlib import Path
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
PARTICIPANT_DIR = Path(r"C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\participant_data")
OUTPUTS_DIR     = BASE_DIR / "outputs"
TIMEPOINTS_CSV  = BASE_DIR / "user_study_timepoints.csv"
OUT_DIR         = BASE_DIR / "analysis_results_2" / "vlm_proxy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Ground truth & metadata ───────────────────────────────────────────────────
GROUND_TRUTH = {
    "amb_d_drawer_close": "A",
    "amb_l_block":        "A",
    "amb_r_block":        "B",
    "amb_to_drawer_close":"B",
    "le_d_drawer_close":  "A",
    "le_l_block":         "A",
    "le_r_block":         "B",
    "le_t_drawer_close":  "B",
}
TRAJ_TYPE = {k: "Ambiguous" if k.startswith("amb") else "Legible"
             for k in GROUND_TRUTH}

VIDEO_DISPLAY = {
    "amb_d_drawer_close":  "Amb\nBot-Drawer",
    "amb_l_block":         "Amb\nL-Block",
    "amb_r_block":         "Amb\nR-Block",
    "amb_to_drawer_close": "Amb\nTop-Drawer",
    "le_d_drawer_close":   "Leg\nBot-Drawer",
    "le_l_block":          "Leg\nL-Block",
    "le_r_block":          "Leg\nR-Block",
    "le_t_drawer_close":   "Leg\nTop-Drawer",
}
VIDEOS = list(GROUND_TRUTH.keys())

# Participant anonymization map (sorted alphabetically; non-ASCII → P8)
ANON_MAP = {
    "Abhi": "P1", "Kartikay": "P2", "Prabhath": "P3", "Raj": "P4",
    "Ryan": "P5", "Sho": "P6", "Summu": "P7",
}
def _anon(name: str) -> str:
    if isinstance(name, str) and name.isascii():
        return ANON_MAP.get(name, name)
    return "P8"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "figure.dpi":    150,
})

C_HUMAN = "#1E40AF"   # deep blue
C_VLM   = "#B45309"   # amber/orange
C_AMB   = "#6D28D9"   # purple for ambiguous
C_LEG   = "#059669"   # green for legible
C_GRID  = "#E5E7EB"


# ============================================================================
# 1. LOAD HUMAN DATA
# ============================================================================

def load_human_data() -> pd.DataFrame:
    """Load all participant JSON files into one tidy DataFrame."""
    VIDEO_ID_MAP = {
        "amb_d_drawer_cclose": "amb_d_drawer_close",
        "amb d drawer close":  "amb_d_drawer_close",
    }
    rows = []
    for jf in sorted(PARTICIPANT_DIR.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            for obs in data:
                vid = obs.get("video_id", "")
                vid = VIDEO_ID_MAP.get(vid, vid)
                if vid not in GROUND_TRUTH:
                    continue
                rows.append({
                    "participant":  _anon(obs.get("participant_id", "")),
                    "video_id":     vid,
                    "timepoint":    int(obs.get("timepoint", 0)),
                    "choice":       str(obs.get("choice", "C")).upper(),
                    "confidence":   obs.get("confidence_0_10"),
                })
        except Exception as e:
            print(f"  Warning: {jf.name}: {e}")
    df = pd.DataFrame(rows)
    df["gt"]      = df["video_id"].map(GROUND_TRUTH)
    df["correct"] = (df["choice"] == df["gt"]).astype(int)
    print(f"Loaded {len(df)} human observations from {df['participant'].nunique()} participants "
          f"over {df['video_id'].nunique()} videos")
    return df


# ============================================================================
# 2. COMPUTE PER-PARTICIPANT PER-TRAJECTORY ACCURACY
# ============================================================================

def human_scores_per_trajectory(human_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (participant, video) pair:
      accuracy = fraction of timepoints where choice == ground truth.
    Returns a (participant × video) accuracy matrix.
    """
    # pivot: participant × video → mean accuracy across timepoints
    pp = (human_df
          .groupby(["participant", "video_id"])["correct"]
          .mean()
          .reset_index()
          .rename(columns={"correct": "accuracy"}))
    pp["accuracy_pct"] = pp["accuracy"] * 100
    return pp


def aggregate_human_by_trajectory(pp: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD human accuracy per trajectory across participants."""
    agg = (pp.groupby("video_id")["accuracy_pct"]
             .agg(human_mean="mean", human_sd="std", human_n="count")
             .reset_index())
    agg["human_se"] = agg["human_sd"] / np.sqrt(agg["human_n"])
    agg["traj_type"] = agg["video_id"].map(TRAJ_TYPE)
    return agg


# ============================================================================
# 3. LOAD VLM DATA (prefix mode, study timepoints only)
# ============================================================================

def load_study_timepoints() -> dict:
    df = pd.read_csv(TIMEPOINTS_CSV)
    tps = {}
    for _, row in df.iterrows():
        tps[row["video_id"]] = [int(t) for t in str(row["recommended_timepoints"]).split(",")]
    return tps


def load_vlm_data(study_timepoints: dict) -> pd.DataFrame:
    """
    Load VLM prefix-mode results for each video, keeping only the timepoints
    that were shown to human participants.  
    When multiple prefix runs exist for a video, use the latest one.
    """
    rows = []
    for vid in VIDEOS:
        vid_dir = OUTPUTS_DIR / vid
        if not vid_dir.exists():
            print(f"  Warning: no outputs folder for {vid}")
            continue
        prefix_runs = sorted(vid_dir.glob("*prefix*/results.jsonl"))
        if not prefix_runs:
            print(f"  Warning: no prefix run for {vid}")
            continue
        # Use the most recent run
        run_file = prefix_runs[-1]
        study_tps = set(study_timepoints.get(vid, []))
        with open(run_file) as f:
            for line in f:
                item = json.loads(line)
                t = item.get("t_sec", item.get("timepoint", -1))
                if t not in study_tps:
                    continue
                gt = GROUND_TRUTH.get(vid)
                choice = str(item.get("choice", "C")).upper()
                # pA is probability assigned to goal A; pScore is probability for correct goal
                pA = float(item.get("pA", 0.5))
                pB = float(item.get("pB", 0.5))
                p_correct = pA if gt == "A" else pB
                rows.append({
                    "video_id":   vid,
                    "timepoint":  t,
                    "choice":     choice,
                    "gt":         gt,
                    "correct":    int(choice == gt),
                    "pA":         pA,
                    "pB":         pB,
                    "p_correct":  p_correct,
                })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} VLM predictions (filtered to study timepoints) "
          f"across {df['video_id'].nunique()} videos")
    return df


def vlm_scores_per_trajectory(vlm_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate VLM: accuracy % and mean p_correct per trajectory."""
    agg = (vlm_df.groupby("video_id")
                 .agg(
                     vlm_accuracy=("correct", lambda x: x.mean() * 100),
                     vlm_p_correct=("p_correct", "mean"),
                     vlm_n=("correct", "count"),
                 )
                 .reset_index())
    return agg


# ============================================================================
# 4. MERGE AND COMPUTE SPEARMAN r
# ============================================================================

def merge_scores(human_agg: pd.DataFrame, vlm_agg: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(human_agg, vlm_agg, on="video_id", how="inner")
    df = df.set_index("video_id").loc[VIDEOS].reset_index()
    return df


def spearman_analysis(merged: pd.DataFrame) -> dict:
    """Compute Spearman correlations: human mean vs VLM accuracy & VLM p_correct."""
    n = len(merged)
    # --- Correlation 1: human accuracy vs VLM binary accuracy
    r1, p1 = stats.spearmanr(merged["human_mean"], merged["vlm_accuracy"])
    # --- Correlation 2: human accuracy vs VLM continuous p_correct
    r2, p2 = stats.spearmanr(merged["human_mean"], merged["vlm_p_correct"])
    # --- Critical r for two-tailed alpha=0.05, n=8
    # t = r * sqrt(n-2) / sqrt(1-r^2), df = n-2 = 6
    t_crit = stats.t.ppf(0.975, df=n - 2)
    r_crit = t_crit / np.sqrt(t_crit**2 + (n - 2))

    results = {
        "n_trajectories": int(n),
        "r_crit_alpha05": round(float(r_crit), 4),
        "spearman_R_accuracy": {
            "r": round(float(r1), 4), "p": round(float(p1), 4),
            "significant": bool(p1 < 0.05),
            "interpretation": "Significant" if p1 < 0.05 else "Trend (not significant)"
        },
        "spearman_R_p_correct": {
            "r": round(float(r2), 4), "p": round(float(p2), 4),
            "significant": bool(p2 < 0.05),
            "interpretation": "Significant" if p2 < 0.05 else "Trend (not significant)"
        },
    }
    return results


# ============================================================================
# 5. FIGURES
# ============================================================================

def fig_bar_human_vs_vlm(merged: pd.DataFrame, out_dir: Path):
    """
    Figure 1: Dumbbell / connected-dot chart — human mean ± SD (circle) vs
    VLM accuracy (square) per trajectory.
    Replaces dynamite-plot bars to show within-group variance honestly.
    """
    # Sort ascending so highest human accuracy is at the top of the chart
    df = merged.sort_values("human_mean", ascending=True).reset_index(drop=True)
    y  = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, row in df.iterrows():
        col = C_AMB if TRAJ_TYPE[row["video_id"]] == "Ambiguous" else C_LEG
        hm  = row["human_mean"]
        hsd = row["human_sd"]
        vv  = row["vlm_accuracy"]
        yi  = float(y[i])

        # ±1 SD shaded range behind human mean
        ax.plot([hm - hsd, hm + hsd], [yi, yi],
                color=col, lw=8, alpha=0.15, solid_capstyle="round", zorder=1)
        # Connecting line human → VLM
        ax.plot([hm, vv], [yi, yi],
                color=col, lw=1.5, alpha=0.5,
                ls="--" if vv < hm else "-", zorder=2)
        # Human mean dot
        ax.scatter(hm, yi, s=130, color=C_HUMAN, zorder=5,
                   edgecolors="white", linewidths=1.3)
        # VLM square marker
        ax.scatter(vv, yi, s=130, color=C_VLM, marker="s", zorder=5,
                   edgecolors="white", linewidths=1.3)

    ax.axvline(50, color="grey", lw=1.0, ls="--", alpha=0.5, zorder=0)
    ax.text(50.5, len(y) - 0.3, "Chance (50%)",
            fontsize=8, color="grey", style="italic", va="top")

    labels = [VIDEO_DISPLAY[row["video_id"]].replace("\n", " ")
              for _, row in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    for tick, (_, row) in zip(ax.get_yticklabels(), df.iterrows()):
        tick.set_color(C_AMB if TRAJ_TYPE[row["video_id"]] == "Ambiguous" else C_LEG)

    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(15, 100)
    ax.set_title(
        "Human vs VLM Goal Inference Accuracy per Trajectory\n"
        "(● = human mean;  bar = ±1 SD;  ■ = VLM;  - - = VLM below human)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.yaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    leg_handles = [
        mlines.Line2D([0], [0], color=C_HUMAN, marker="o", ms=9, ls="none",
                      markeredgecolor="white", markeredgewidth=1.3,
                      label="Human mean (±1 SD shaded)"),
        mlines.Line2D([0], [0], color=C_VLM, marker="s", ms=9, ls="none",
                      markeredgecolor="white", markeredgewidth=1.3,
                      label="VLM (Gemini)"),
        mpatches.Patch(color=C_AMB, alpha=0.75, label="Ambiguous trajectory"),
        mpatches.Patch(color=C_LEG, alpha=0.75, label="Legible trajectory"),
    ]
    ax.legend(handles=leg_handles, fontsize=10, framealpha=0.9, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_human_vs_vlm_bar.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig1_human_vs_vlm_bar.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig1_human_vs_vlm_bar.png/pdf")


def fig_scatter_spearman(merged: pd.DataFrame, corr_results: dict, out_dir: Path):
    """
    Figure 2: Scatter plot — mean human accuracy vs VLM accuracy.
    Annotated with Spearman r and p.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (vlm_col, label, c_col) in zip(
        axes,
        [("vlm_accuracy",  "VLM Binary Accuracy (%)",          "spearman_R_accuracy"),
         ("vlm_p_correct", "VLM Mean P(correct goal)", "spearman_R_p_correct")],
    ):
        y = merged[vlm_col].values
        x = merged["human_mean"].values

        # Color by trajectory type
        colors = [C_AMB if TRAJ_TYPE[v] == "Ambiguous" else C_LEG
                  for v in merged["video_id"]]
        ax.scatter(x, y, c=colors, s=120, zorder=4, edgecolors="#1F2937", linewidths=0.8)

        # Regression line (OLS, for visual only — reported stat is Spearman)
        m, b, *_ = stats.linregress(x, y)
        xline = np.linspace(x.min() - 5, x.max() + 5, 100)
        ax.plot(xline, m * xline + b, color="#6B7280", lw=1.5, ls="--", zorder=3, alpha=0.7)

        # Labels on each point
        for xi, yi, vid in zip(x, y, merged["video_id"]):
            short = VIDEO_DISPLAY[vid].replace("\n", " ")
            ax.annotate(short, (xi, yi), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, color="#374151")

        # Spearman annotation box
        r = corr_results[c_col]["r"]
        p = corr_results[c_col]["p"]
        sig_str = "p < 0.05 ✓" if p < 0.05 else f"p = {p:.3f}"
        annot = f"Spearman r = {r:.3f}\n{sig_str}\nn = {corr_results['n_trajectories']} trajectories"
        ax.text(0.05, 0.95, annot, transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F9FAFB",
                          edgecolor="#D1D5DB", linewidth=1.2))

        ax.set_xlabel("Mean Human Accuracy (%)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.yaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
        ax.xaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)

    # Shared legend
    leg_handles = [
        mpatches.Patch(color=C_AMB, label="Ambiguous trajectory"),
        mpatches.Patch(color=C_LEG, label="Legible trajectory"),
    ]
    axes[0].legend(handles=leg_handles, fontsize=10, loc="lower right", framealpha=0.9)
    fig.suptitle("Spearman Correlation: Human Mean Accuracy vs VLM Score",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_scatter_spearman.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig2_scatter_spearman.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig2_scatter_spearman.png/pdf")


def fig_participant_heatmap(pp: pd.DataFrame, out_dir: Path):
    """
    Figure 3: Heatmap of per-participant per-trajectory accuracy (%).
    Shows the raw distribution behind the mean ± SD.
    """
    matrix = (pp.pivot(index="participant", columns="video_id", values="accuracy_pct")
               .reindex(columns=VIDEOS))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    cax = ax.imshow(matrix.values, aspect="auto", cmap="RdYlGn",
                    vmin=0, vmax=100, zorder=2)

    # Annotations: value in each cell
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if (val < 30 or val > 75) else "#1F2937")

    # Labels
    ax.set_xticks(range(len(VIDEOS)))
    ax.set_xticklabels([VIDEO_DISPLAY[v].replace("\n", "\n") for v in VIDEOS],
                       fontsize=9)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=10)

    # Color x-labels by type
    for tick, vid in zip(ax.get_xticklabels(), VIDEOS):
        tick.set_color(C_AMB if TRAJ_TYPE[vid] == "Ambiguous" else C_LEG)

    cb = fig.colorbar(cax, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Accuracy (%)", fontsize=10)
    ax.set_title("Per-Participant Accuracy per Trajectory (% correct timepoints)",
                 fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_participant_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_participant_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig3_participant_heatmap.png/pdf")


def fig_summary_table(merged: pd.DataFrame, corr_results: dict, out_dir: Path):
    """
    Figure 4: Clean summary table (mean ± SD, VLM acc, Spearman r) as a figure
    suitable for embedding in a thesis/paper.
    """
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")

    col_labels = ["Trajectory", "Type",
                  "Human\nMean ± SD (%)",
                  "VLM\nAccuracy (%)",
                  "VLM\np(correct)"]
    rows = []
    for _, row in merged.iterrows():
        rows.append([
            row["video_id"].replace("_", " "),
            TRAJ_TYPE[row["video_id"]],
            f"{row['human_mean']:.1f} ± {row['human_sd']:.1f}",
            f"{row['vlm_accuracy']:.1f}",
            f"{row['vlm_p_correct']:.3f}",
        ])

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.7)

    # Header styling
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#1E3A5F")
        cell.set_text_props(color="white", fontweight="bold")

    # Row colors by type
    for i, row in enumerate(rows):
        is_amb = row[1] == "Ambiguous"
        bg = "#EDE9FE" if is_amb else "#D1FAE5"
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)

    # Spearman footnote
    r1 = corr_results["spearman_R_accuracy"]["r"]
    p1 = corr_results["spearman_R_accuracy"]["p"]
    r2 = corr_results["spearman_R_p_correct"]["r"]
    p2 = corr_results["spearman_R_p_correct"]["p"]
    rcrit = corr_results["r_crit_alpha05"]
    note = (f"Spearman r (human mean vs VLM accuracy) = {r1:.3f},  p = {p1:.3f}   |   "
            f"Spearman r (human mean vs VLM p_correct) = {r2:.3f},  p = {p2:.3f}   |   "
            f"Critical r (n=8, α=0.05) = {rcrit:.3f}")
    ax.text(0.5, 0.01, note, ha="center", va="bottom",
            transform=ax.transAxes, fontsize=9, color="#374151",
            style="italic")

    ax.set_title("VLM vs Human Performance Summary (n = 8 trajectories, 8 participants)",
                 fontsize=13, fontweight="bold", pad=8)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_summary_table.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_summary_table.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig4_summary_table.png/pdf")


# ============================================================================
# 6. SAVE RESULTS TO CSV / JSON
# ============================================================================

def save_results(merged: pd.DataFrame, corr_results: dict, pp: pd.DataFrame,
                 out_dir: Path):
    # Main trajectory-level table
    cols = ["video_id", "traj_type", "human_mean", "human_sd", "human_se",
            "human_n", "vlm_accuracy", "vlm_p_correct", "vlm_n"]
    merged[cols].to_csv(out_dir / "trajectory_scores.csv", index=False, float_format="%.4f")

    # Per-participant per-trajectory
    pp[["participant", "video_id", "accuracy_pct"]].sort_values(
        ["video_id", "participant"]
    ).to_csv(out_dir / "per_participant_accuracy.csv", index=False, float_format="%.2f")

    # Correlation results
    with open(out_dir / "spearman_results.json", "w") as f:
        json.dump(corr_results, f, indent=2)

    print(f"\n  Saved CSVs and JSON to {out_dir}")


# ============================================================================
# 7. PRINT REPORT
# ============================================================================

def print_report(merged: pd.DataFrame, corr_results: dict):
    print("\n" + "=" * 70)
    print("VLM AS PROXY FOR HUMAN JUDGMENT — ANALYSIS REPORT")
    print("=" * 70)
    print(f"\nn = {corr_results['n_trajectories']} trajectories  |  "
          f"Critical r (two-tailed a=0.05) = {corr_results['r_crit_alpha05']}")

    print("\n── Per-Trajectory Scores ──────────────────────────────────────────")
    print(f"{'Trajectory':<25}  {'Type':<10}  {'Human mean±SD':>15}  "
          f"{'VLM acc':>8}  {'VLM p_corr':>10}")
    print("-" * 75)
    for _, row in merged.iterrows():
        print(f"{row['video_id']:<25}  {row['traj_type']:<10}  "
              f"{row['human_mean']:>6.1f} ± {row['human_sd']:>5.1f}%  "
              f"{row['vlm_accuracy']:>8.1f}%  {row['vlm_p_correct']:>10.3f}")

    print("\n── Spearman Correlations ──────────────────────────────────────────")
    for key, label in [("spearman_R_accuracy", "Human mean accuracy vs VLM binary accuracy"),
                        ("spearman_R_p_correct", "Human mean accuracy vs VLM p(correct)")]:
        res = corr_results[key]
        flag = " ✓ SIGNIFICANT" if res["significant"] else " (trend, not sig.)"
        print(f"\n  {label}")
        print(f"    r = {res['r']:+.4f},  p = {res['p']:.4f}{flag}")
        print(f"    {res['interpretation']}")

    print("\n── Interpretation ────────────────────────────────────────────────")
    r1 = corr_results["spearman_R_accuracy"]["r"]
    r2 = corr_results["spearman_R_p_correct"]["r"]
    rcrit = corr_results["r_crit_alpha05"]
    if max(abs(r1), abs(r2)) >= rcrit:
        print("  VLM scores are statistically significantly correlated with human")
        print("  judgment — VLM can serve as a proxy for human legibility assessment.")
    else:
        print(f"  Neither correlation reached significance (|r| < {rcrit}).")
        print("  With n=8 trajectories, power is low. Report the correlation as a trend.")
        print("  Collect more trajectory stimuli to increase power.")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    study_tps    = load_study_timepoints()
    human_df     = load_human_data()
    pp           = human_scores_per_trajectory(human_df)
    human_agg    = aggregate_human_by_trajectory(pp)
    vlm_df       = load_vlm_data(study_tps)
    vlm_agg      = vlm_scores_per_trajectory(vlm_df)
    merged       = merge_scores(human_agg, vlm_agg)
    corr_results = spearman_analysis(merged)

    print_report(merged, corr_results)

    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    fig_bar_human_vs_vlm(merged, OUT_DIR)
    fig_scatter_spearman(merged, corr_results, OUT_DIR)
    fig_participant_heatmap(pp, OUT_DIR)
    fig_summary_table(merged, corr_results, OUT_DIR)

    save_results(merged, corr_results, pp, OUT_DIR)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
