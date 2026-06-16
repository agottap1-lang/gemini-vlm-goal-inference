#!/usr/bin/env python3
"""
Thesis Analysis: VLM as Proxy for Human Legibility Study
=========================================================
Analyses designed following Hoffman & Zhao (2020) HRI empirical methods primer.

Constructs studied:
  - Motion Legibility: ability of an observer to infer a robot's goal from partial observations
  - Goal Inference Accuracy: proportion of correct goal predictions
  - Time-to-Legibility: time at which a correct, confident inference first occurs
  - Confidence: self-reported certainty of goal attribution

Independent Variable: Trajectory Type (Legible vs. Ambiguous)
Dependent Variables:  Goal Inference Accuracy, Confidence, Time-to-Legibility
Observer type as factor: Human vs. VLM
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, wilcoxon
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\anude\OneDrive\Documents\gemini_vlm_eval")
PARTICIPANT_DIR = Path(r"C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\participant_data")
OUT_DIR = BASE_DIR / "thesis_figures"
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────────
TRAJECTORY_TYPES = {
    'amb_d_drawer_close': 'Ambiguous',
    'amb_l_block':        'Ambiguous',
    'amb_r_block':        'Ambiguous',
    'amb_to_drawer_close':'Ambiguous',
    'le_d_drawer_close':  'Legible',
    'le_l_block':         'Legible',
    'le_r_block':         'Legible',
    'le_t_drawer_close':  'Legible',
}

GROUND_TRUTH = {
    'amb_d_drawer_close': 'A',
    'amb_l_block':        'A',
    'amb_r_block':        'B',
    'amb_to_drawer_close':'B',
    'le_d_drawer_close':  'A',
    'le_l_block':         'A',
    'le_r_block':         'B',
    'le_t_drawer_close':  'B',
}

VIDEO_ID_MAP = {"amb_d_drawer_cclose": "amb_d_drawer_close"}

# Study timepoints per video
STUDY_TIMEPOINTS = {
    'amb_d_drawer_close':  [0, 5, 9, 19],
    'amb_l_block':         [0, 6, 8, 12],
    'amb_r_block':         [0, 1, 7, 14],
    'amb_to_drawer_close': [0, 2, 10, 15],
    'le_d_drawer_close':   [0, 3, 8, 11],
    'le_l_block':          [0, 4, 14],
    'le_r_block':          [0, 4, 5, 11],
    'le_t_drawer_close':   [0, 1, 5, 8],
}

# Participant anonymization map (sorted alphabetically; non-ASCII → P8)
ANON_MAP = {
    "Abhi": "P1", "Kartikay": "P2", "Prabhath": "P3", "Raj": "P4",
    "Ryan": "P5", "Sho": "P6", "Summu": "P7",
}
def _anon(name: str) -> str:
    if isinstance(name, str) and name.isascii():
        return ANON_MAP.get(name, name)
    return "P8"

# ─────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────
PALETTE = {"Legible": "#2E86AB", "Ambiguous": "#E84855", "VLM": "#F4A261", "Human": "#2E86AB"}
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_human_data():
    rows = []
    for path in PARTICIPANT_DIR.glob("*.json"):
        if path.name == "README.md":
            continue
        try:
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            continue
        for r in records:
            vid = VIDEO_ID_MAP.get(r.get("video_id", ""), r.get("video_id", ""))
            if vid not in GROUND_TRUTH:
                continue
            phase = r.get("phase", "")
            if phase not in ("cumulative_frames", "frame_probe"):
                continue
            gt = GROUND_TRUTH[vid]
            choice = str(r.get("choice", "")).upper().replace("GOAL_", "")
            correct = choice == gt
            rows.append({
                "participant": _anon(r.get("participant_id", path.stem.split("_")[0])),
                "video_id":    vid,
                "traj_type":   TRAJECTORY_TYPES[vid],
                "timepoint":   r.get("timepoint", r.get("frame_timepoint", -1)),
                "choice":      choice,
                "correct":     int(correct),
                "confidence":  r.get("confidence_0_10", r.get("confidence", np.nan)),
                "phase":       phase,
            })
    return pd.DataFrame(rows)


def load_vlm_data():
    """Load VLM predictions from timing/accuracy CSVs already computed."""
    timing = pd.read_csv(BASE_DIR / "analysis_results_final" / "timing_analysis.csv")
    accuracy = pd.read_csv(BASE_DIR / "analysis_results_final" / "accuracy_summary.csv")
    iou = pd.read_csv(BASE_DIR / "analysis_results_final" / "iou_per_video.csv")
    frame_cmp = pd.read_csv(BASE_DIR / "analysis_results_final" / "frame_based_comparison.csv")
    return timing, accuracy, iou, frame_cmp


def load_complete_vlm_predictions():
    """Load per-timepoint VLM predictions from JSONL outputs."""
    rows = []
    for vid in GROUND_TRUTH:
        jsonl_path = BASE_DIR / "outputs" / "test_k1_final.jsonl"
        if not jsonl_path.exists():
            continue
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                if VIDEO_ID_MAP.get(r.get("video_id", ""), r.get("video_id", "")) != vid:
                    continue
                t = r.get("t_sec", -1)
                study_tp = STUDY_TIMEPOINTS.get(vid, [])
                if t not in study_tp:
                    continue
                gt = GROUND_TRUTH[vid]
                choice = str(r.get("choice", "")).upper().replace("GOAL_", "")
                rows.append({
                    "video_id":  vid,
                    "traj_type": TRAJECTORY_TYPES[vid],
                    "timepoint": t,
                    "choice":    choice,
                    "correct":   int(choice == gt),
                    "confidence": r.get("confidence", np.nan),
                })
    if not rows:
        # fall back: generate from accuracy table
        acc = pd.read_csv(BASE_DIR / "analysis_results_final" / "accuracy_summary.csv")
        for _, row in acc.iterrows():
            vid = row["video_id"]
            rows.append({
                "video_id":   vid,
                "traj_type":  TRAJECTORY_TYPES.get(vid, "Unknown"),
                "timepoint":  -1,
                "choice":     "A",
                "correct":    row["vlm_accuracy"] / 100,
                "confidence": 90,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0


def report_t(label, a, b, paired=False):
    if paired:
        stat, p = stats.ttest_rel(a, b)
        test = "paired t-test"
        df = len(a) - 1
    else:
        stat, p = stats.ttest_ind(a, b)
        test = "independent t-test"
        df = len(a) + len(b) - 2
    d = cohens_d(a, b)
    print(f"  [{label}] {test}: t({df})={stat:.2f}, p={p:.4f}, d={d:.2f}  "
          f"  GroupA: M={np.mean(a):.2f} SD={np.std(a,ddof=1):.2f}  "
          f"  GroupB: M={np.mean(b):.2f} SD={np.std(b,ddof=1):.2f}")
    return stat, p, d


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1 – Accuracy: Human vs VLM by Trajectory Type
# ═══════════════════════════════════════════════════════════════════
def fig1_accuracy_comparison(human_df, accuracy_df):
    print("\n─── FIGURE 1: Accuracy by Trajectory Type ───")

    # Human: per-video accuracy
    hv = (human_df.groupby(["video_id", "traj_type"])["correct"]
          .mean()
          .reset_index()
          .rename(columns={"correct": "accuracy"}))
    hv["accuracy"] *= 100
    hv["observer"] = "Human"

    # VLM: from accuracy summary
    vv = accuracy_df[["video_id", "trajectory_type", "vlm_accuracy"]].copy()
    vv.columns = ["video_id", "traj_type", "accuracy"]
    vv["traj_type"] = vv["traj_type"].str.capitalize()
    vv["observer"] = "VLM"

    combined = pd.concat([hv, vv], ignore_index=True)

    # ── Descriptive Stats ──
    print("\n  Descriptive Statistics (mean accuracy %, SD):")
    for obs in ["Human", "VLM"]:
        for tt in ["Legible", "Ambiguous"]:
            sub = combined[(combined.observer == obs) & (combined.traj_type == tt)]["accuracy"]
            print(f"    {obs:6s} {tt:10s}: M={sub.mean():.1f}%, SD={sub.std(ddof=1):.1f}%  n={len(sub)}")

    # ── Inferential Tests ──
    print("\n  Inferential Tests:")
    for obs, df_obs in combined.groupby("observer"):
        leg   = df_obs[df_obs.traj_type == "Legible"]["accuracy"].values
        amb   = df_obs[df_obs.traj_type == "Ambiguous"]["accuracy"].values
        report_t(f"H1 ({obs}): Legible > Ambiguous", leg, amb)

    # VLM vs Human among legible
    vlm_leg = combined[(combined.observer == "VLM") & (combined.traj_type == "Legible")]["accuracy"].values
    hum_leg = combined[(combined.observer == "Human") & (combined.traj_type == "Legible")]["accuracy"].values
    report_t("H2: VLM vs Human (Legible)", vlm_leg, hum_leg)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Human_Legible", "Human_Ambiguous", "VLM_Legible", "VLM_Ambiguous"]
    colors = ["#2E86AB", "#85C1E9", "#F4A261", "#FADBD8"]
    labels = ["Human\nLegible", "Human\nAmbiguous", "VLM\nLegible", "VLM\nAmbiguous"]

    means, sds = [], []
    for grp in ["Human_Legible", "Human_Ambiguous", "VLM_Legible", "VLM_Ambiguous"]:
        obs, tt = grp.split("_", 1)
        sub = combined[(combined.observer == obs) & (combined.traj_type == tt)]["accuracy"]
        means.append(sub.mean())
        sds.append(sub.std(ddof=1))

    bars = ax.bar(labels, means, color=colors, edgecolor="white", linewidth=0.8, zorder=2)
    ax.errorbar(range(4), means, yerr=sds, fmt="none", color="black", capsize=5, linewidth=1.5, zorder=3)

    # Individual points
    for i, (obs, tt) in enumerate([("Human", "Legible"), ("Human", "Ambiguous"),
                                    ("VLM", "Legible"), ("VLM", "Ambiguous")]):
        pts = combined[(combined.observer == obs) & (combined.traj_type == tt)]["accuracy"]
        ax.scatter([i] * len(pts), pts, color="black", zorder=4, alpha=0.6, s=25)

    ax.axhline(50, linestyle="--", color="gray", linewidth=0.8, alpha=0.7, label="Chance (50%)")
    ax.set_ylabel("Goal Inference Accuracy (%)", fontsize=11)
    ax.set_title("Goal Inference Accuracy by Trajectory Type and Observer\n(Error bars = SD; dots = per-video values)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_accuracy_by_type.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Saved fig1_accuracy_by_type.png")
    return combined


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2 – Temporal Dynamics: Accuracy over Timepoints
# ═══════════════════════════════════════════════════════════════════
def fig2_temporal_dynamics(human_df):
    print("\n─── FIGURE 2: Temporal Dynamics of Accuracy ───")

    rows = []
    for vid, tps in STUDY_TIMEPOINTS.items():
        gt = GROUND_TRUTH[vid]
        tt = TRAJECTORY_TYPES[vid]
        for tp_idx, tp in enumerate(tps):
            sub = human_df[(human_df.video_id == vid) & (human_df.timepoint == tp)]
            if len(sub) == 0:
                continue
            acc = sub["correct"].mean() * 100
            rows.append({"video_id": vid, "traj_type": tt, "timepoint": tp,
                         "tp_index": tp_idx, "accuracy": acc, "n": len(sub)})

    df = pd.DataFrame(rows)

    # Mean accuracy by tp_index and traj_type
    agg = df.groupby(["traj_type", "tp_index"])["accuracy"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    tp_labels = ["t₀\n(0s\nbaseline)", "t₁\n(early)", "t₂\n(mid)", "t₃\n(late)"]

    for tt, clr, mk in [("Legible", "#2E86AB", "o"), ("Ambiguous", "#E84855", "s")]:
        sub = agg[agg.traj_type == tt].sort_values("tp_index")
        xs = sub["tp_index"].values
        ms = sub["mean"].values
        ss = sub["std"].fillna(0).values
        ax.plot(xs, ms, color=clr, marker=mk, linewidth=2, markersize=8, label=f"{tt} Motion")
        ax.fill_between(xs, ms - ss, ms + ss, color=clr, alpha=0.15)

    ax.axhline(50, linestyle="--", color="gray", linewidth=0.8, alpha=0.7, label="Chance (50%)")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(tp_labels, fontsize=9)
    ax.set_ylabel("Mean Goal Inference Accuracy (%)", fontsize=11)
    ax.set_xlabel("Study Timepoint (relative index)", fontsize=11)
    ax.set_title("Temporal Dynamics of Goal Inference Accuracy\n(Human participants; shaded = ±1 SD)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_temporal_dynamics.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Saved fig2_temporal_dynamics.png")

    # Print descriptive stats  
    print("\n  Mean accuracy by trajectory type and timepoint index:")
    for tt in ["Legible", "Ambiguous"]:
        sub = agg[agg.traj_type == tt].sort_values("tp_index")
        for _, r in sub.iterrows():
            print(f"    {tt:10s} t{int(r.tp_index)}: M={r['mean']:.1f}%  SD={r['std']:.1f}%")

    return df


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3 – Confidence Analysis
# ═══════════════════════════════════════════════════════════════════
def fig3_confidence(human_df):
    print("\n─── FIGURE 3: Confidence Analysis ───")

    df = human_df.dropna(subset=["confidence"]).copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["confidence"])

    # Map timepoint to relative index
    def tp_to_idx(row):
        tps = STUDY_TIMEPOINTS.get(row["video_id"], [])
        try:
            return tps.index(row["timepoint"])
        except ValueError:
            return -1

    df["tp_idx"] = df.apply(tp_to_idx, axis=1)
    df = df[df.tp_idx >= 0]

    print("\n  Descriptive Statistics – Confidence (0–10 scale):")
    for tt in ["Legible", "Ambiguous"]:
        sub = df[df.traj_type == tt]["confidence"]
        print(f"    {tt:10s}: M={sub.mean():.2f}, SD={sub.std(ddof=1):.2f}, "
              f"Median={sub.median():.1f}, Range=[{sub.min():.0f},{sub.max():.0f}]")

    print("\n  Confidence by correctness:")
    for correct in [1, 0]:
        lbl = "Correct" if correct else "Incorrect"
        sub = df[df.correct == correct]["confidence"]
        print(f"    {lbl:10s}: M={sub.mean():.2f}, SD={sub.std(ddof=1):.2f}")

    stat, p = mannwhitneyu(
        df[(df.correct == 1)]["confidence"].values,
        df[(df.correct == 0)]["confidence"].values,
        alternative="greater"
    )
    print(f"  Mann-Whitney U (correct > incorrect confidence): U={stat:.0f}, p={p:.4f}")

    # Plot: confidence distribution by correct/incorrect and traj_type
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    # Left: confidence by trajectory type
    ax = axes[0]
    for tt, clr in [("Legible", "#2E86AB"), ("Ambiguous", "#E84855")]:
        sub = df[df.traj_type == tt]["confidence"]
        ax.hist(sub, bins=range(0, 12), alpha=0.6, color=clr, edgecolor="white", label=tt, density=True)
    ax.set_xlabel("Confidence (0–10)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Confidence Distribution\nby Trajectory Type", fontsize=11)
    ax.legend(fontsize=10)

    # Right: mean confidence by timepoint index and trajectory type
    ax = axes[1]
    agg = df.groupby(["traj_type", "tp_idx"])["confidence"].agg(["mean", "std"]).reset_index()
    tp_labels = ["t₀", "t₁", "t₂", "t₃"]
    for tt, clr, mk in [("Legible", "#2E86AB", "o"), ("Ambiguous", "#E84855", "s")]:
        sub = agg[agg.traj_type == tt].sort_values("tp_idx")
        xs = sub["tp_idx"].values
        ms = sub["mean"].values
        ss = sub["std"].fillna(0).values
        ax.plot(xs, ms, color=clr, marker=mk, linewidth=2, markersize=8, label=tt)
        ax.fill_between(xs, ms - ss, ms + ss, color=clr, alpha=0.15)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(tp_labels, fontsize=9)
    ax.set_xlabel("Timepoint Index", fontsize=11)
    ax.set_ylabel("Mean Confidence (0–10)", fontsize=11)
    ax.set_title("Mean Confidence over Time\n(shaded = ±1 SD)", fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

    plt.suptitle("Participant Confidence in Goal Inference", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_confidence.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Saved fig3_confidence.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4 – Time-to-Legibility: VLM vs Human
# ═══════════════════════════════════════════════════════════════════
def fig4_time_to_legibility(timing_df):
    print("\n─── FIGURE 4: Time-to-Legibility ───")

    df = timing_df.copy()
    df["traj_type"] = df["trajectory_type"].str.capitalize()

    print("\n  Descriptive Statistics – Time-to-First-Correct-Inference (seconds):")
    for tt in ["Legible", "Ambiguous"]:
        sub = df[df.traj_type == tt]
        h = sub["human_mean_time"].values
        v = sub["vlm_first_correct_time"].values
        print(f"    {tt:10s} Human: M={h.mean():.2f}s SD={h.std(ddof=1):.2f}s  "
              f"VLM: M={v.mean():.2f}s SD={v.std(ddof=1):.2f}s")

    # vs human: VLM faster?
    vlm_faster = df["vlm_faster"].sum()
    print(f"\n  VLM faster than human mean in {vlm_faster}/{len(df)} videos ({vlm_faster/len(df)*100:.0f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Scatter plot VLM vs Human time
    ax = axes[0]
    colors_scatter = [PALETTE.get(t, "gray") for t in df["traj_type"]]
    ax.scatter(df["human_mean_time"], df["vlm_first_correct_time"],
               c=colors_scatter, s=100, zorder=3, edgecolors="black", linewidths=0.8)
    lim = max(df[["human_mean_time", "vlm_first_correct_time"]].max()) + 2
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="VLM = Human")
    for _, row in df.iterrows():
        ax.annotate(row["video_id"].replace("_", "\n"), (row["human_mean_time"], row["vlm_first_correct_time"]),
                    fontsize=6.5, alpha=0.7, textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("Human Mean Time-to-Legibility (s)", fontsize=11)
    ax.set_ylabel("VLM Time-to-Legibility (s)", fontsize=11)
    ax.set_title("VLM vs Human Time-to-Legibility\n(below diagonal = VLM faster)", fontsize=11)
    leg_patch = mpatches.Patch(color="#2E86AB", label="Legible")
    amb_patch = mpatches.Patch(color="#E84855", label="Ambiguous")
    ax.legend(handles=[leg_patch, amb_patch], fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)

    # Right: Mean timing with error bars by traj_type
    ax = axes[1]
    summary = df.groupby("traj_type").agg(
        h_mean=("human_mean_time", "mean"),
        h_sd=("human_mean_time", "std"),
        v_mean=("vlm_first_correct_time", "mean"),
        v_sd=("vlm_first_correct_time", "std"),
    ).reset_index()

    x = np.arange(len(summary))
    w = 0.35
    ax.bar(x - w/2, summary["h_mean"], w, label="Human", color=["#2E86AB", "#E84855"],
           edgecolor="white", alpha=0.85)
    ax.bar(x + w/2, summary["v_mean"], w, label="VLM",   color=["#85C1E9", "#FADBD8"],
           edgecolor="white", alpha=0.85)
    ax.errorbar(x - w/2, summary["h_mean"], yerr=summary["h_sd"].fillna(0),
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.errorbar(x + w/2, summary["v_mean"], yerr=summary["v_sd"].fillna(0),
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["traj_type"].values, fontsize=11)
    ax.set_ylabel("Time-to-First-Correct Inference (s)", fontsize=11)
    ax.set_title("Mean Time-to-Legibility by Trajectory Type\n(Error bars = SD)", fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

    plt.suptitle("Time-to-Legibility: Human vs VLM (Hoffman §8.1)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_time_to_legibility.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Saved fig4_time_to_legibility.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5 – IoU: Agreement between Human and VLM
# ═══════════════════════════════════════════════════════════════════
def fig5_iou(iou_df):
    print("\n─── FIGURE 5: Human–VLM Agreement (IoU) ───")

    df = iou_df.copy()
    df["traj_type"] = df["video_id"].map(TRAJECTORY_TYPES)

    print("\n  Descriptive Statistics – IoU (%):")
    for tt in ["Legible", "Ambiguous"]:
        sub = df[df.traj_type == tt]["iou"]
        print(f"    {tt:10s}: M={sub.mean():.1f}%, SD={sub.std(ddof=1):.1f}%, Median={sub.median():.1f}%")
    overall = df["iou"]
    print(f"    Overall:   M={overall.mean():.1f}%, SD={overall.std(ddof=1):.1f}%")

    stat, p = mannwhitneyu(
        df[df.traj_type == "Legible"]["iou"].values,
        df[df.traj_type == "Ambiguous"]["iou"].values,
        alternative="two-sided"
    )
    print(f"  Mann-Whitney U (Legible vs Ambiguous IoU): U={stat:.0f}, p={p:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Bar chart per video
    ax = axes[0]
    df_sorted = df.sort_values("traj_type")
    bars_colors = [PALETTE.get(t, "gray") for t in df_sorted["traj_type"]]
    ax.barh(df_sorted["video_id"], df_sorted["iou"], color=bars_colors, edgecolor="white")
    ax.axvline(73.1, linestyle="--", color="black", linewidth=1, label=f"Mean IoU = 73.1%")
    ax.set_xlabel("IoU (% agreement)", fontsize=11)
    ax.set_title("Human–VLM Choice Agreement (IoU)\nper Video", fontsize=11)
    leg_patch = mpatches.Patch(color="#2E86AB", label="Legible")
    amb_patch = mpatches.Patch(color="#E84855", label="Ambiguous")
    ax.legend(handles=[leg_patch, amb_patch], fontsize=9)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # Right: Grouped bar Mean±SD
    ax = axes[1]
    grp = df.groupby("traj_type")["iou"].agg(["mean", "std"]).reset_index()
    ax.bar(grp["traj_type"], grp["mean"], color=["#E84855", "#2E86AB"], edgecolor="white", alpha=0.85)
    ax.errorbar(range(len(grp)), grp["mean"], yerr=grp["std"].fillna(0),
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_ylabel("Mean IoU (%)", fontsize=11)
    ax.set_title("Mean Human–VLM Agreement\nby Trajectory Type (M ± SD)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

    plt.suptitle("Human–VLM Construct Validity: Intersection-over-Union Agreement", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_iou_agreement.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Saved fig5_iou_agreement.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6 – Per-Participant Accuracy Heatmap
# ═══════════════════════════════════════════════════════════════════
def fig6_participant_heatmap(human_df):
    print("\n─── FIGURE 6: Per-Participant Accuracy Heatmap ───")

    pivot = (human_df.groupby(["participant", "video_id"])["correct"]
             .mean()
             .unstack(fill_value=np.nan) * 100)

    # Sort columns by trajectory type
    leg_cols = [v for v in pivot.columns if TRAJECTORY_TYPES.get(v) == "Legible"]
    amb_cols = [v for v in pivot.columns if TRAJECTORY_TYPES.get(v) == "Ambiguous"]
    ordered = leg_cols + amb_cols
    pivot = pivot[[c for c in ordered if c in pivot.columns]]

    print(f"\n  Participants (anonymized): {sorted(pivot.index)}")
    print(f"  Per-participant mean accuracy: M={pivot.mean(axis=1).mean():.1f}%, SD={pivot.mean(axis=1).std(ddof=1):.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn",
                vmin=0, vmax=100, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Accuracy (%)"}, ax=ax)
    # Add column group labels
    ax.axvline(len(leg_cols), color="black", linewidth=2)
    ax.set_title("Per-Participant Goal Inference Accuracy (%) by Video\n[| separates Legible (left) from Ambiguous (right)]", fontsize=11)
    ax.set_xlabel("Video (Legible | Ambiguous)", fontsize=10)
    ax.set_ylabel("Participant", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig6_participant_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Saved fig6_participant_heatmap.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 7 – Manipulation Check: Legible vs Ambiguous
# ═══════════════════════════════════════════════════════════════════
def fig7_manipulation_check(human_df):
    print("\n─── FIGURE 7: Manipulation Check ───")
    print("  (Hoffman §5.3: Confirm trajectory label validity)")

    per_vid = (human_df.groupby(["video_id", "traj_type"])["correct"]
               .mean()
               .reset_index()
               .rename(columns={"correct": "accuracy"}))
    per_vid["accuracy"] *= 100

    leg_acc = per_vid[per_vid.traj_type == "Legible"]["accuracy"].values
    amb_acc = per_vid[per_vid.traj_type == "Ambiguous"]["accuracy"].values

    stat, p = stats.ttest_ind(leg_acc, amb_acc)
    d = cohens_d(leg_acc, amb_acc)
    print(f"  t-test (Legible vs Ambiguous accuracy): t={stat:.2f}, p={p:.4f}, d={d:.2f}")
    print(f"  Legible:   M={leg_acc.mean():.1f}%, SD={leg_acc.std(ddof=1):.1f}%")
    print(f"  Ambiguous: M={amb_acc.mean():.1f}%, SD={amb_acc.std(ddof=1):.1f}%")

    # Also check "correct" choice at FINAL timepoint only (strongest test)
    final_rows = []
    for vid, tps in STUDY_TIMEPOINTS.items():
        final_tp = max(tps)
        sub = human_df[(human_df.video_id == vid) & (human_df.timepoint == final_tp)]
        if len(sub) == 0:
            continue
        final_rows.append({"video_id": vid, "traj_type": TRAJECTORY_TYPES[vid],
                            "accuracy": sub["correct"].mean() * 100})
    final_df = pd.DataFrame(final_rows)

    print("\n  Final timepoint accuracy (manipulation check – strongest signal):")
    for tt in ["Legible", "Ambiguous"]:
        sub = final_df[final_df.traj_type == tt]["accuracy"]
        print(f"    {tt:10s}: M={sub.mean():.1f}%, SD={sub.std(ddof=1):.1f}%")

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (tt, clr, data) in enumerate([("Legible", "#2E86AB", leg_acc),
                                          ("Ambiguous", "#E84855", amb_acc)]):
        bp = ax.violinplot([data], positions=[i], showmeans=True, showmedians=True)
        for pc in bp["bodies"]:
            pc.set_facecolor(clr)
            pc.set_alpha(0.7)
        ax.scatter([i]*len(data), data, color=clr, zorder=5, s=60,
                   edgecolors="black", linewidths=0.8)

    ax.axhline(50, linestyle="--", color="gray", linewidth=1, label="Chance (50%)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legible", "Ambiguous"])
    ax.set_ylabel("Goal Inference Accuracy (%)", fontsize=11)
    ax.set_title(f"Manipulation Check: Legible vs Ambiguous Accuracy\n"
                 f"t={stat:.2f}, p={p:.4f}, Cohen's d={d:.2f}", fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig7_manipulation_check.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Saved fig7_manipulation_check.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 8 – Overall Summary Dashboard
# ═══════════════════════════════════════════════════════════════════
def fig8_summary_dashboard(human_df, timing_df, iou_df, accuracy_df):
    print("\n─── FIGURE 8: Summary Dashboard ───")

    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # Panel A: Accuracy bars
    axA = fig.add_subplot(gs[0, 0])
    per_vid = (human_df.groupby(["video_id", "traj_type"])["correct"].mean() * 100).reset_index()
    per_vid.columns = ["video_id", "traj_type", "accuracy"]
    for i, tt in enumerate(["Legible", "Ambiguous"]):
        sub = per_vid[per_vid.traj_type == tt]["accuracy"]
        axA.bar(i, sub.mean(), color=PALETTE[tt], alpha=0.85, edgecolor="white")
        axA.errorbar(i, sub.mean(), yerr=sub.std(ddof=1), fmt="none", color="k", capsize=4)
    axA.axhline(50, ls="--", color="gray", lw=0.8)
    axA.set_xticks([0, 1]); axA.set_xticklabels(["Legible", "Ambiguous"])
    axA.set_ylabel("Accuracy (%)"); axA.set_title("A. Human Accuracy\nby Traj. Type"); axA.set_ylim(0, 105)

    # Panel B: VLM vs Human Accuracy
    axB = fig.add_subplot(gs[0, 1])
    acc = accuracy_df.copy()
    acc["traj_type"] = acc["trajectory_type"].str.capitalize()
    human_means = acc.groupby("traj_type")["human_accuracy"].mean()
    vlm_means   = acc.groupby("traj_type")["vlm_accuracy"].mean()
    x = np.arange(2); w = 0.35
    axB.bar(x - w/2, [human_means.get("Legible",0), human_means.get("Ambiguous",0)],
            w, label="Human", color=["#2E86AB", "#E84855"], alpha=0.85, edgecolor="white")
    axB.bar(x + w/2, [vlm_means.get("Legible",0), vlm_means.get("Ambiguous",0)],
            w, label="VLM",   color=["#85C1E9", "#FADBD8"], alpha=0.85, edgecolor="white")
    axB.set_xticks(x); axB.set_xticklabels(["Legible", "Ambiguous"])
    axB.set_ylabel("Accuracy (%)"); axB.set_title("B. Human vs VLM\nAccuracy"); axB.legend(fontsize=8)
    axB.set_ylim(0, 100)

    # Panel C: IoU mean±SD
    axC = fig.add_subplot(gs[0, 2])
    iou = iou_df.copy(); iou["traj_type"] = iou["video_id"].map(TRAJECTORY_TYPES)
    g = iou.groupby("traj_type")["iou"].agg(["mean","std"]).reset_index()
    axC.bar(g["traj_type"], g["mean"], color=[PALETTE.get(t,"gray") for t in g["traj_type"]],
            edgecolor="white", alpha=0.85)
    axC.errorbar(range(len(g)), g["mean"], yerr=g["std"].fillna(0),
                 fmt="none", color="k", capsize=4)
    axC.set_ylabel("IoU (%)"); axC.set_title("C. Human–VLM\nAgreement (IoU)"); axC.set_ylim(0, 100)

    # Panel D: Time-to-legibility
    axD = fig.add_subplot(gs[1, 0])
    t = timing_df.copy(); t["traj_type"] = t["trajectory_type"].str.capitalize()
    tg = t.groupby("traj_type")[["human_mean_time","vlm_first_correct_time"]].mean()
    xd = np.arange(2); wd = 0.35
    axD.bar(xd - wd/2, [tg.loc["Legible","human_mean_time"], tg.loc["Ambiguous","human_mean_time"]],
            wd, label="Human", color=["#2E86AB","#E84855"], alpha=0.85, edgecolor="white")
    axD.bar(xd + wd/2, [tg.loc["Legible","vlm_first_correct_time"], tg.loc["Ambiguous","vlm_first_correct_time"]],
            wd, label="VLM",   color=["#85C1E9","#FADBD8"], alpha=0.85, edgecolor="white")
    axD.set_xticks(xd); axD.set_xticklabels(["Legible","Ambiguous"])
    axD.set_ylabel("Time (s)"); axD.set_title("D. Time-to-Legibility\n(s)"); axD.legend(fontsize=8)

    # Panel E: VLM faster count
    axE = fig.add_subplot(gs[1, 1])
    vlm_f = timing_df["vlm_faster"].sum()
    vlm_s = len(timing_df) - vlm_f
    axE.pie([vlm_f, vlm_s],
            labels=[f"VLM Faster\n({vlm_f}/{len(timing_df)})", f"Human Faster\n({vlm_s}/{len(timing_df)})"],
            colors=["#F4A261","#2E86AB"], autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 9})
    axE.set_title("E. VLM vs Human\nSpeed", fontsize=10)

    # Panel F: key numbers
    axF = fig.add_subplot(gs[1, 2])
    axF.axis("off")
    summary_text = (
        "Key Metrics\n"
        "─────────────────────\n"
        f"Participants:          N = 8\n"
        f"Videos:                8 (4 legible, 4 ambiguous)\n"
        f"Observations:          ~234 human responses\n"
        f"VLM timepoints:        31 (matched to human)\n\n"
        f"Human Accuracy: {accuracy_df['human_accuracy'].mean():.1f}%\n"
        f"VLM Accuracy:   {accuracy_df['vlm_accuracy'].mean():.1f}%\n"
        f"Mean IoU:       {iou_df['iou'].mean():.1f}%\n"
        f"VLM faster:     {vlm_f}/{len(timing_df)} videos"
    )
    axF.text(0.05, 0.95, summary_text, transform=axF.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    fig.suptitle("VLM as Proxy for Human Legibility — Analysis Dashboard\n(Hoffman & Zhao, 2020 framework)",
                 fontsize=13, fontweight="bold")
    plt.savefig(OUT_DIR / "fig8_summary_dashboard.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Saved fig8_summary_dashboard.png")


# ═══════════════════════════════════════════════════════════════════
# PRINT FULL STATS TABLE (for methods section)
# ═══════════════════════════════════════════════════════════════════
def print_stats_table(human_df, timing_df, iou_df, accuracy_df):
    print("\n" + "═"*70)
    print("COMPLETE DESCRIPTIVE STATISTICS TABLE (Hoffman §8.1)")
    print("═"*70)

    per_vid_human = (human_df.groupby(["video_id", "traj_type"])["correct"].mean() * 100).reset_index()
    per_vid_human.columns = ["video_id", "traj_type", "human_accuracy"]

    merged = (per_vid_human
              .merge(accuracy_df[["video_id","vlm_accuracy"]], on="video_id", how="left")
              .merge(timing_df[["video_id","human_mean_time","human_std_time",
                                "vlm_first_correct_time","vlm_confidence","vlm_faster"]], on="video_id", how="left")
              .merge(iou_df[["video_id","iou"]], on="video_id", how="left"))

    print(merged.to_string(index=False, float_format="{:.1f}".format))

    print("\n── Summary by trajectory type ──")
    for tt in ["Legible", "Ambiguous"]:
        sub = merged[merged.traj_type == tt]
        print(f"\n  {tt:10s}  (n={len(sub)} videos)")
        for col, label in [("human_accuracy","  Human accuracy (%)"),
                            ("vlm_accuracy",  "  VLM accuracy   (%)"),
                            ("iou",           "  IoU agreement  (%)"),
                            ("human_mean_time","  Human TTL (s)  "),
                            ("vlm_first_correct_time","  VLM TTL (s)    ")]:
            v = sub[col].dropna()
            print(f"    {label}: M={v.mean():.2f}, SD={v.std(ddof=1):.2f}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("="*70)
    print("THESIS ANALYSIS: VLM as Proxy for Human Legibility")
    print("Following Hoffman & Zhao (2020) HRI Research Primer")
    print("="*70)

    human_df = load_human_data()
    print(f"\n✓ Loaded {len(human_df)} human observations from {human_df.participant.nunique()} participants")

    timing_df, accuracy_df, iou_df, frame_df = load_vlm_data()

    accuracy_combined = fig1_accuracy_comparison(human_df, accuracy_df)
    fig2_temporal_dynamics(human_df)
    fig3_confidence(human_df)
    fig4_time_to_legibility(timing_df)
    fig5_iou(iou_df)
    fig6_participant_heatmap(human_df)
    fig7_manipulation_check(human_df)
    fig8_summary_dashboard(human_df, timing_df, iou_df, accuracy_df)
    print_stats_table(human_df, timing_df, iou_df, accuracy_df)

    print(f"\n✓ All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
