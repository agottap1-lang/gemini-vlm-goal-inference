#!/usr/bin/env python3
"""
Part 1 human-study vs VLM comparison — all three providers.

Loads:
  - Human Part-1 data (phase == 'cumulative_frames', 31 timepoints, 8 participants)
  - Gemini 3 Pro predictions  (analysis_3/results_gemini_3_pro_preview.jsonl)
  - GPT-4o predictions        (outputs/results_gpt4o.jsonl)
  - Claude Opus predictions   (outputs/results_claude_opus.jsonl)

Aligns VLM predictions to the 31 human timepoints.

Outputs (to --out folder):
  - figure_summary_table.png/.pdf   — per-video table (mirrors figure4 style)
  - figure_accuracy_bar.png/.pdf    — grouped bar chart: human + 3 VLMs
  - figure_agreement_bar.png/.pdf   — human-VLM agreement per video + provider
  - part1_comparison.csv            — raw numbers

Accuracy definition (identical to original study):
  - ALL choices count in the denominator (A, B, and C).
  - Only exact matches to ground truth count in the numerator.
  - C = "not legible yet" is treated as WRONG — it never matches A or B.
  - This "C-as-wrong" rule matches the original study (overall ~66.7%).

Committed accuracy (supplementary metric):
  - Only committed choices (A or B) count in numerator or denominator.
  - C observations are excluded entirely.
  - Answers: "when the human/VLM actually committed, how often were they right?"

Agreement / IoU definition:
  - For each (video, timepoint): compare every human choice against the VLM
    choice at that same (video, timepoint).
  - Agreement = # human choices that match VLM choice / total human choices.
    (Both A, B, and C count toward the denominator here — same as figure4.)
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows (avoids UnicodeEncodeError for emoji/arrows)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARTICIPANT_DATA_DIR = Path(
    r"C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\participant_data"
)

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

TRAJ_TYPE = {
    "amb_d_drawer_close": "Ambiguous",
    "amb_l_block":        "Ambiguous",
    "amb_r_block":        "Ambiguous",
    "amb_to_drawer_close":"Ambiguous",
    "le_d_drawer_close":  "Legible",
    "le_l_block":         "Legible",
    "le_r_block":         "Legible",
    "le_t_drawer_close":  "Legible",
}

VID_LABEL = {
    "amb_d_drawer_close":  "Amb: Bottom Drawer",
    "amb_l_block":         "Amb: Left Block",
    "amb_r_block":         "Amb: Right Block",
    "amb_to_drawer_close": "Amb: Top Drawer",
    "le_d_drawer_close":   "Leg: Bottom Drawer",
    "le_l_block":          "Leg: Left Block",
    "le_r_block":          "Leg: Right Block",
    "le_t_drawer_close":   "Leg: Top Drawer",
}

PROVIDER_COLORS = {
    "Human (Part 1)":    "#1B4332",   # dark green  (displayed as "Reference")
    "Gemini 2.5 Flash":  "#34A853",   # Google green
    "Gemini 3 Pro":      "#4285F4",   # Google blue
    "GPT-4o":            "#EA4335",   # red
    "GPT-5.4":           "#C0392B",   # dark red
    "Claude Opus 4.5":   "#FBBC05",   # yellow
    "Claude Opus 4-6":   "#E67E22",   # orange
}

VIDEO_ID_MAPPING = {
    "amb d drawer close":  "amb_d_drawer_close",
    "amb_d_drawer_cclose": "amb_d_drawer_close",   # typo in participant data
    "amb d drawer cclose": "amb_d_drawer_close",   # typo (space variant)
    "amb l block":         "amb_l_block",
    "amb r block":         "amb_r_block",
    "amb to drawer close": "amb_to_drawer_close",
    "le d drawer close":   "le_d_drawer_close",
    "le l block":          "le_l_block",
    "le r block":          "le_r_block",
    "le t drawer close":   "le_t_drawer_close",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_human_part1() -> pd.DataFrame:
    """Load only Part-1 observations (phase == 'cumulative_frames')."""
    rows = []
    for json_file in sorted(PARTICIPANT_DATA_DIR.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for obs in (data if isinstance(data, list) else []):
                if not isinstance(obs, dict):
                    continue
                if obs.get("phase") != "cumulative_frames":
                    continue
                vid = VIDEO_ID_MAPPING.get(obs.get("video_id", ""), obs.get("video_id", ""))
                if vid not in GROUND_TRUTH:
                    continue
                rows.append({
                    "participant_id": obs.get("participant_id", "?"),
                    "video_id":       vid,
                    "timepoint":      int(obs.get("timepoint", 0)),
                    "choice":         str(obs.get("choice", "C")).upper(),
                    "goal_gt":        GROUND_TRUTH[vid],
                })
        except Exception as e:
            print(f"  ⚠️  {json_file.name}: {e}")

    df = pd.DataFrame(rows)
    print(f"Human Part-1 loaded: {len(df)} observations, "
          f"{df['participant_id'].nunique()} participants, "
          f"{df['video_id'].nunique()} videos")
    return df


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
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
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def human_accuracy_per_video(human_df: pd.DataFrame) -> dict:
    """Overall accuracy: correct / ALL choices  (C counts as wrong).

    Denominator = every observation (A, B, and C).
    Numerator   = observations where choice == ground truth (A or B).
    C is never equal to ground truth, so it is implicitly penalised.
    This rule reproduces the original study's ~66.7% overall accuracy.
    """
    out = {}
    for vid, grp in human_df.groupby("video_id"):
        if len(grp) == 0:
            out[vid] = 0.0
        else:
            out[vid] = (grp["choice"] == grp["goal_gt"]).mean() * 100
    overall = (human_df["choice"] == human_df["goal_gt"]).mean() * 100
    return {"per_video": out, "overall": overall}


def human_committed_accuracy_per_video(human_df: pd.DataFrame) -> dict:
    """Committed accuracy: correct / (A+B choices only, C excluded).

    Only rows where the participant chose A or B enter the calculation.
    Answers: 'when humans actually committed to a goal, how often were they right?'
    """
    committed = human_df[human_df["choice"].isin(["A", "B"])].copy()
    out = {}
    for vid, grp in committed.groupby("video_id"):
        out[vid] = (grp["choice"] == grp["goal_gt"]).mean() * 100
    overall = (committed["choice"] == committed["goal_gt"]).mean() * 100 if len(committed) > 0 else 0.0
    n_committed = len(committed)
    n_total = len(human_df)
    return {"per_video": out, "overall": overall,
            "n_committed": n_committed, "n_total": n_total}


def vlm_accuracy_per_video(vlm_df: pd.DataFrame, shared_keys: set) -> dict:
    """Overall VLM accuracy at shared timepoints — C counts as wrong (same rule as human).

    Denominator = every aligned observation (A, B, and C).
    Numerator   = observations where VLM choice == ground truth.
    C is never equal to ground truth → penalised, matching human accuracy rule.
    """
    aligned = vlm_df[
        vlm_df.apply(lambda r: (r["video_id"], int(r["t_sec"])) in shared_keys, axis=1)
    ].copy()

    out = {}
    for vid, grp in aligned.groupby("video_id"):
        if len(grp) == 0:
            out[vid] = 0.0
        else:
            out[vid] = (grp["choice"] == grp["goal_gt"]).mean() * 100

    overall = (aligned["choice"] == aligned["goal_gt"]).mean() * 100 if len(aligned) > 0 else 0.0
    n_abstain = (aligned["choice"] == "C").sum()

    # Also compute committed accuracy (C excluded)
    committed = aligned[aligned["choice"].isin(["A", "B"])].copy()
    comm_out = {}
    for vid, grp in committed.groupby("video_id"):
        comm_out[vid] = (grp["choice"] == grp["goal_gt"]).mean() * 100
    comm_overall = (committed["choice"] == committed["goal_gt"]).mean() * 100 if len(committed) > 0 else 0.0

    return {
        "per_video": out,
        "overall": overall,
        "n_abstain": int(n_abstain),
        "n_total": len(aligned),
        "committed_per_video": comm_out,
        "committed_overall": comm_overall,
        "n_committed": len(committed),
    }


def agreement_per_video(human_df: pd.DataFrame, vlm_df: pd.DataFrame,
                        shared_keys: set) -> dict:
    """
    For each (video, timepoint) in shared_keys, compare every human choice
    against the single VLM choice.  Agreement = matches / total comparisons.
    All choices (A, B, C) count toward the denominator.
    """
    vlm_lookup = {}
    for _, row in vlm_df.iterrows():
        key = (row["video_id"], int(row["t_sec"]))
        if key in shared_keys:
            vlm_lookup[key] = row["choice"]

    out = {}
    total_matches = 0
    total_count   = 0

    for vid in GROUND_TRUTH:
        h_vid = human_df[human_df["video_id"] == vid]
        matches = 0
        count   = 0
        for tp in h_vid["timepoint"].unique():
            if (vid, tp) not in vlm_lookup:
                continue
            vlm_choice = vlm_lookup[(vid, tp)]
            h_choices  = h_vid[h_vid["timepoint"] == tp]["choice"].tolist()
            for hc in h_choices:
                count += 1
                if hc == vlm_choice:
                    matches += 1
        out[vid] = (matches / count * 100) if count > 0 else 0.0
        total_matches += matches
        total_count   += count

    overall = (total_matches / total_count * 100) if total_count > 0 else 0.0
    return {"per_video": out, "overall": overall}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, stem: Path) -> None:
    """Save figure as PNG + PDF.  Resize PNG to max 7900px per side to stay
    within the Anthropic/Copilot 8000-pixel-per-dimension upload limit."""
    for ext in (".png", ".pdf"):
        fig.savefig(str(stem) + ext, bbox_inches="tight", dpi=150)
    plt.close(fig)
    # Post-process PNG: cap any dimension at 7900px
    png_path = str(stem) + ".png"
    try:
        from PIL import Image as _PILImage
        img = _PILImage.open(png_path)
        w, h = img.size
        if w > 7900 or h > 7900:
            scale = 7900 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), _PILImage.LANCZOS)
            img.save(png_path)
            print(f"  Resized PNG {w}x{h} → {new_w}x{new_h}: {png_path}")
    except ImportError:
        pass  # Pillow not installed — skip resize


def fig_human_choice_breakdown(human_df: pd.DataFrame, out_stem: Path) -> None:
    """Per-video stacked bar chart: Correct / Wrong / Uncertain (C) choices at each timepoint.

    Each subplot = one video.  X-axis = timepoints shown to humans.
    Stacked bars show how many participants (out of 8) chose:
      - Correct (green)  : choice == ground truth (A or B)
      - Wrong   (red)    : choice != ground truth AND choice in {A, B}
      - Uncertain (grey) : choice == C
    """
    leg_vids = sorted([v for v, t in TRAJ_TYPE.items() if t == "Legible"])
    amb_vids = sorted([v for v, t in TRAJ_TYPE.items() if t == "Ambiguous"])
    all_vids = leg_vids + amb_vids

    CORRECT_COLOR   = "#2D9E4E"
    WRONG_COLOR     = "#E53E3E"
    UNCERTAIN_COLOR = "#A0AEC0"

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    axes_flat = axes.flatten()

    for ax_idx, vid in enumerate(all_vids):
        ax = axes_flat[ax_idx]
        vid_df = human_df[human_df["video_id"] == vid]
        timepoints = sorted(vid_df["timepoint"].unique())
        gt = GROUND_TRUTH[vid]

        correct_counts   = []
        wrong_counts     = []
        uncertain_counts = []
        total_counts     = []

        for tp in timepoints:
            tp_df = vid_df[vid_df["timepoint"] == tp]
            n_correct   = (tp_df["choice"] == gt).sum()
            n_uncertain = (tp_df["choice"] == "C").sum()
            n_wrong     = len(tp_df) - n_correct - n_uncertain
            correct_counts.append(n_correct)
            wrong_counts.append(n_wrong)
            uncertain_counts.append(n_uncertain)
            total_counts.append(len(tp_df))

        x = np.arange(len(timepoints))
        bar_w = 0.55

        b_correct   = ax.bar(x, correct_counts,   bar_w,
                              color=CORRECT_COLOR,   label="Correct",   zorder=3)
        b_wrong     = ax.bar(x, wrong_counts,     bar_w,
                              bottom=correct_counts,
                              color=WRONG_COLOR,     label="Wrong",     zorder=3)
        b_uncertain = ax.bar(x, uncertain_counts, bar_w,
                              bottom=[c + w for c, w in zip(correct_counts, wrong_counts)],
                              color=UNCERTAIN_COLOR, label="Uncertain (C)", zorder=3, alpha=0.8)

        # Annotate total bar count
        for i, (c, w, u) in enumerate(zip(correct_counts, wrong_counts, uncertain_counts)):
            total = c + w + u
            ax.text(i, total + 0.15, str(total), ha="center", va="bottom",
                    fontsize=7.5, color="black")

        # Accuracy % annotations at top of Correct segment
        for i, c in enumerate(correct_counts):
            total = total_counts[i]
            if total > 0:
                pct = c / total * 100
                ax.text(i, c / 2, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"t={tp}s" for tp in timepoints], fontsize=8)
        ax.set_ylim(0, 10)
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        ttype = TRAJ_TYPE[vid]
        bg_color = "#EFF6FF" if ttype == "Legible" else "#FEF2F2"
        ax.set_facecolor(bg_color)

        short_name = VID_LABEL[vid].split(": ")[1]
        ax.set_title(f"[{ttype[0]}] {short_name}\n(GT={gt})",
                     fontsize=9, fontweight="bold")

    # Shared y-label
    for ax in axes[:, 0]:
        ax.set_ylabel("# Participants", fontsize=9)

    # Shared legend
    correct_patch   = mpatches.Patch(color=CORRECT_COLOR,   label="Correct (=GT)")
    wrong_patch     = mpatches.Patch(color=WRONG_COLOR,     label="Wrong (≠GT)")
    uncertain_patch = mpatches.Patch(color=UNCERTAIN_COLOR, label="Uncertain (C)", alpha=0.8)
    fig.legend(handles=[correct_patch, wrong_patch, uncertain_patch],
               loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.01), frameon=True)

    fig.suptitle(
        "Human Part-1 Choices per Video & Timepoint\n"
        "(Correct=green, Wrong=red, Uncertain C=grey  |  % inside bar = overall accuracy incl. C)\n"
        "Blue bg = Legible, Red bg = Ambiguous",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, out_stem)
    print(f"  Saved: {out_stem}.*")


def fig_summary_table(human_acc: dict, providers: dict, out_stem: Path) -> None:
    """Reproduce figure4-style summary table with all providers."""

    leg_vids = [v for v, t in TRAJ_TYPE.items() if t == "Legible"]
    amb_vids = [v for v, t in TRAJ_TYPE.items() if t == "Ambiguous"]
    all_vids  = sorted(leg_vids) + sorted(amb_vids)

    prov_names = list(providers.keys())  # e.g. ["Gemini 3 Pro", "GPT-4o", "Claude Opus"]

    # Build column labels and rows
    col_labels = ["Video", "Type", "Human\nAcc (Part 1)"]
    for name in prov_names:
        col_labels += [f"{name}\nAcc", f"{name}\nAgreement"]
    col_labels += ["Mean\nAgreement"]

    table_rows = []
    for v in all_vids:
        short = VID_LABEL[v].split(": ")[1]
        row = [short, TRAJ_TYPE[v], f"{human_acc['per_video'].get(v, 0):.1f}%"]
        agr_vals = []
        for name in prov_names:
            acc_v = providers[name]["accuracy"]["per_video"].get(v, 0)
            agr_v = providers[name]["agreement"]["per_video"].get(v, 0)
            row.append(f"{acc_v:.1f}%")
            row.append(f"{agr_v:.1f}%")
            agr_vals.append(agr_v)
        row.append(f"{np.mean(agr_vals):.1f}%")
        table_rows.append(row)

    # Mean row
    mean_row = ["Mean", "—", f"{human_acc['overall']:.1f}%"]
    for name in prov_names:
        mean_row.append(f"{providers[name]['accuracy']['overall']:.1f}%")
        mean_row.append(f"{providers[name]['agreement']['overall']:.1f}%")
    means_agr = [providers[n]["agreement"]["overall"] for n in prov_names]
    mean_row.append(f"{np.mean(means_agr):.1f}%")
    table_rows.append(mean_row)

    n_cols = len(col_labels)
    fig_w  = max(14, n_cols * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    ax.axis("off")
    tbl = ax.table(cellText=table_rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    # Header styling
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#1F2937")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Row shading by type
    for i, v in enumerate(all_vids, start=1):
        bg = "#EFF6FF" if TRAJ_TYPE[v] == "Legible" else "#FEF2F2"
        for j in range(n_cols):
            tbl[(i, j)].set_facecolor(bg)

    # Mean row bold + grey
    last = len(table_rows)
    for j in range(n_cols):
        tbl[(last, j)].set_facecolor("#F3F4F6")
        tbl[(last, j)].set_text_props(fontweight="bold")

    ax.set_title(
        "Part-1 Study: Human vs VLM — Goal Inference Accuracy & Agreement per Video\n"
        "(blue = Legible; red = Ambiguous | Accuracy: C counts as wrong for all, matching original study)",
        fontsize=10, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    save_fig(fig, out_stem)
    print(f"  Saved: {out_stem}.*")


def fig_summary_table_committed(human_comm_acc: dict, providers: dict, out_stem: Path) -> None:
    """Summary table using COMMITTED accuracy (C excluded from denominator).

    Agreement column is identical to the C-as-wrong table because the current
    agreement metric already counts C==C as a match (all choices A/B/C compare
    equally against the VLM's choice at each timepoint).

    Note printed in the table subtitle makes this explicit.
    """
    leg_vids = [v for v, t in TRAJ_TYPE.items() if t == "Legible"]
    amb_vids = [v for v, t in TRAJ_TYPE.items() if t == "Ambiguous"]
    all_vids  = sorted(leg_vids) + sorted(amb_vids)

    prov_names = list(providers.keys())

    col_labels = ["Video", "Type", "Human\nComm. Acc"]
    for name in prov_names:
        col_labels += [f"{name}\nComm. Acc", f"{name}\nAgreement*"]
    col_labels += ["Mean\nAgreement*"]

    table_rows = []
    for v in all_vids:
        short = VID_LABEL[v].split(": ")[1]
        row = [short, TRAJ_TYPE[v], f"{human_comm_acc['per_video'].get(v, 0):.1f}%"]
        agr_vals = []
        for name in prov_names:
            comm_acc_v = providers[name]["accuracy"]["committed_per_video"].get(v, 0)
            agr_v      = providers[name]["agreement"]["per_video"].get(v, 0)
            row.append(f"{comm_acc_v:.1f}%")
            row.append(f"{agr_v:.1f}%")
            agr_vals.append(agr_v)
        row.append(f"{np.mean(agr_vals):.1f}%")
        table_rows.append(row)

    # Mean row
    mean_row = ["Mean", "\u2014", f"{human_comm_acc['overall']:.1f}%"]
    for name in prov_names:
        mean_row.append(f"{providers[name]['accuracy']['committed_overall']:.1f}%")
        mean_row.append(f"{providers[name]['agreement']['overall']:.1f}%")
    means_agr = [providers[n]["agreement"]["overall"] for n in prov_names]
    mean_row.append(f"{np.mean(means_agr):.1f}%")
    table_rows.append(mean_row)

    n_cols = len(col_labels)
    fig_w  = max(14, n_cols * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    ax.axis("off")
    tbl = ax.table(cellText=table_rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#14532D")   # dark green header to distinguish
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    for i, v in enumerate(all_vids, start=1):
        bg = "#EFF6FF" if TRAJ_TYPE[v] == "Legible" else "#FEF2F2"
        for j in range(n_cols):
            tbl[(i, j)].set_facecolor(bg)

    last = len(table_rows)
    for j in range(n_cols):
        tbl[(last, j)].set_facecolor("#F3F4F6")
        tbl[(last, j)].set_text_props(fontweight="bold")

    ax.set_title(
        "Part-1 Study: Committed Accuracy (C excluded) & Agreement per Video\n"
        "(blue=Legible; red=Ambiguous | Comm. Acc: C rows removed from denominator)\n"
        "(*Agreement: C==C counts as match -- human & VLM both uncertain = agreement)",
        fontsize=10, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    save_fig(fig, out_stem)
    print(f"  Saved: {out_stem}.*")


# Best-model subset (latest / most capable per provider)
BEST_VLM_LABELS = ["Gemini 3 Pro", "GPT-5.4", "Claude Opus 4.5"]


def fig_heatmap_best(
    human_acc: dict,
    human_comm_acc: dict,
    providers: dict,
    out_stem: Path,
) -> None:
    """Heatmap: per-video accuracy (overall + committed) for best 3 VLMs + Human."""
    leg_vids = sorted(v for v, t in TRAJ_TYPE.items() if t == "Legible")
    amb_vids = sorted(v for v, t in TRAJ_TYPE.items() if t == "Ambiguous")
    row_vids = leg_vids + amb_vids  # Legible on top

    entities = ["Reference\n(Part 1)"] + [
        n.replace(" ", "\n") for n in BEST_VLM_LABELS if n in providers
    ]
    entity_keys = ["Human (Part 1)"] + [n for n in BEST_VLM_LABELS if n in providers]
    col_colors  = [PROVIDER_COLORS[k] for k in entity_keys]

    row_labels = [VID_LABEL[v].split(": ")[1] for v in row_vids]

    # Build two data matrices: overall and committed
    def _build_matrix(vids, acc_key, comm_key):
        mat = np.zeros((len(vids), len(entity_keys)))
        for ri, vid in enumerate(vids):
            for ci, ek in enumerate(entity_keys):
                if ek == "Human (Part 1)":
                    ov  = human_acc["per_video"].get(vid, 0)
                    com = human_comm_acc["per_video"].get(vid, 0)
                else:
                    ov  = providers[ek]["accuracy"]["per_video"].get(vid, 0)
                    com = providers[ek]["accuracy"]["committed_per_video"].get(vid, 0)
                mat[ri, ci] = ov if acc_key == "overall" else com
        return mat

    mat_ov  = _build_matrix(row_vids, "overall",   None)
    mat_com = _build_matrix(row_vids, "committed",  None)

    # Mean row
    mean_ov  = np.array([[human_acc["overall"]] + [
        providers[k]["accuracy"]["overall"] for k in entity_keys[1:]]])
    mean_com = np.array([[human_comm_acc["overall"]] + [
        providers[k]["accuracy"]["committed_overall"] for k in entity_keys[1:]]])

    mat_ov_full  = np.vstack([mat_ov, mean_ov])
    mat_com_full = np.vstack([mat_com, mean_com])
    row_labels_full = row_labels + ["Mean"]

    n_rows = len(row_labels_full)
    n_cols = len(entities)

    fig, axes = plt.subplots(1, 2, figsize=(max(9, n_cols * 2.2), max(6, n_rows * 0.72 + 1.5)),
                              gridspec_kw={"wspace": 0.35})

    cmap = plt.get_cmap("RdYlGn")

    for ax, mat, title in [
        (axes[0], mat_ov_full,  "Accuracy  (C = wrong)"),
        (axes[1], mat_com_full, "Committed Accuracy  (C excluded)"),
    ]:
        im = ax.imshow(mat, vmin=0, vmax=100, cmap=cmap, aspect="auto")

        # Cell text
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = mat[ri, ci]
                txt_color = "white" if (val < 30 or val > 80) else "black"
                ax.text(ci, ri, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=txt_color)

        # Axes labels
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(entities, fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels_full, fontsize=9)

        # Color-code column headers by provider
        for tick, color in zip(ax.get_xticklabels(), col_colors):
            tick.set_color(color)
            tick.set_fontweight("bold")

        # Dividing lines: between Legible/Ambiguous and before Mean
        n_leg = len(leg_vids)
        ax.axhline(n_leg - 0.5, color="white", lw=2.5)
        ax.axhline(n_rows - 1 - 0.5, color="white", lw=2.5)

        # Row background for trajectory type (left margin ticks colored)
        for ri, vid in enumerate(row_vids):
            color = "#1B6CA8" if TRAJ_TYPE[vid] == "Legible" else "#C0392B"
            ax.get_yticklabels()[ri].set_color(color)

        # Section labels on the right
        mid_leg = (n_leg - 1) / 2
        mid_amb = n_leg + (n_rows - 1 - n_leg) / 2 - 0.5
        for side_x in [n_cols - 0.3]:
            ax.text(side_x + 0.6, mid_leg, "Legible",   fontsize=8, color="#1B6CA8",
                    fontweight="bold", va="center", ha="left",
                    transform=ax.transData, rotation=90)
            ax.text(side_x + 0.6, mid_amb, "Ambiguous", fontsize=8, color="#C0392B",
                    fontweight="bold", va="center", ha="left",
                    transform=ax.transData, rotation=90)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#222222")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04,
                     label="Accuracy (%)", shrink=0.82)

    fig.suptitle(
        "Goal Inference Accuracy — Best Models per Provider\n"
        "Blue = Legible · Red = Ambiguous · Green = high accuracy · Red = low",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, out_stem)
    print(f"  Saved: {out_stem}.*")


def fig_cleveland_best(
    human_acc: dict,
    human_comm_acc: dict,
    providers: dict,
    out_stem: Path,
) -> None:
    """Cleveland dot plot: overall vs committed accuracy for best models + human.

    Each entity gets ONE row.  Two dots per row:
      • filled circle = overall accuracy (C counted as wrong)
      • open diamond  = committed accuracy (C excluded)
    A horizontal segment connects the two — the gap shows the cost of abstention.
    """
    entity_keys   = ["Human (Part 1)"] + [n for n in BEST_VLM_LABELS if n in providers]
    entity_labels = ["Reference\n(Part 1)"] + [
        n.replace(" ", "\n") for n in BEST_VLM_LABELS if n in providers
    ]
    colors = [PROVIDER_COLORS[k] for k in entity_keys]

    overall_vals   = []
    committed_vals = []
    for ek in entity_keys:
        if ek == "Human (Part 1)":
            overall_vals.append(human_acc["overall"])
            committed_vals.append(human_comm_acc["overall"])
        else:
            overall_vals.append(providers[ek]["accuracy"]["overall"])
            committed_vals.append(providers[ek]["accuracy"]["committed_overall"])

    # Sort by committed accuracy descending
    order = sorted(range(len(entity_keys)), key=lambda i: committed_vals[i])
    entity_labels  = [entity_labels[i]  for i in order]
    colors         = [colors[i]         for i in order]
    overall_vals   = [overall_vals[i]   for i in order]
    committed_vals = [committed_vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(entity_keys) * 1.05 + 1.3)))

    y_pos = np.arange(len(entity_keys))
    for i, (y, ov, com, color, label) in enumerate(
            zip(y_pos, overall_vals, committed_vals, colors, entity_labels)):
        # Connecting segment
        ax.plot([ov, com], [y, y], color=color, lw=2.2, alpha=0.55, zorder=2)
        # Overall dot (filled circle)
        ax.scatter(ov,  y, color=color, s=110, zorder=4, marker="o",
                   label="Overall (C=wrong)" if i == 0 else "")
        # Committed dot (open diamond)
        ax.scatter(com, y, color=color, s=110, zorder=4, marker="D",
                   facecolors="white", edgecolors=color, linewidths=2,
                   label="Committed (C excl.)" if i == 0 else "")
        # Value labels
        ax.text(ov  - 1.5, y, f"{ov:.1f}%",  ha="right",  va="center",
                fontsize=9,  color=color, fontweight="bold")
        ax.text(com + 1.5, y, f"{com:.1f}%", ha="left",   va="center",
                fontsize=9,  color=color, fontweight="bold")

    # Chance and human-overall reference lines
    ax.axvline(50,  color="grey",     lw=1.0, ls=":",  alpha=0.55, zorder=1)
    ax.text(50.4, -0.55, "Chance\n(50%)", fontsize=7.5, color="grey",
            style="italic", va="top", ha="left")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(entity_labels, fontsize=10.5)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_xlim(0, 115)
    ax.xaxis.grid(True, ls="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="grey", lw=0, markersize=9,
               markerfacecolor="grey", label="Overall accuracy  (C = wrong)"),
        Line2D([0], [0], marker="D", color="grey", lw=0, markersize=9,
               markerfacecolor="white", markeredgecolor="grey", markeredgewidth=2,
               label="Committed accuracy  (C excluded)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right",
              framealpha=0.92, edgecolor="#cccccc")

    ax.set_title(
        "Goal Inference Accuracy — Best Models\n"
        "● Overall (C=wrong)   ◆ Committed (C excluded)   Gap = effect of abstention",
        fontsize=11, fontweight="bold", pad=10,
    )
    fig.tight_layout(pad=1.4)
    save_fig(fig, out_stem)
    print(f"  Saved: {out_stem}.*")


def _draw_clean_table(ax, fig_h, col_headers, col_widths, rows_data,
                      header_bgs, title, footer_text):
    """Shared helper: draws a clean, readable presentation table."""
    n_cols = len(col_headers)
    n_rows = len(rows_data)
    ROW_H  = 0.56
    HEAD_H = 0.64
    total_w = sum(col_widths)

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h - 0.2)
    ax.axis("off")

    xs = [sum(col_widths[:i]) for i in range(n_cols + 1)]
    cx = [(xs[i] + xs[i + 1]) / 2 for i in range(n_cols)]

    TEXT_DARK  = "#111827"
    TEXT_WHITE = "#FFFFFF"
    MEAN_BG    = "#E2E8F0"
    LEG_BG     = "#EFF6FF"
    AMB_BG     = "#FFF1F1"
    GRID_COLOR = "#9CA3AF"

    head_y = fig_h - HEAD_H - 0.18

    # --- header cells (each column gets its own background) ---
    for ci in range(n_cols):
        ax.add_patch(plt.Rectangle((xs[ci], head_y), col_widths[ci], HEAD_H,
                                   fc=header_bgs[ci], ec=GRID_COLOR, lw=0.6, zorder=2))
        ax.text(cx[ci], head_y + HEAD_H / 2, col_headers[ci],
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=TEXT_WHITE, zorder=3)

    # --- data rows ---
    for ri, row in enumerate(rows_data):
        y = head_y - (ri + 1) * ROW_H
        is_mean = (row[0] == "Mean")
        ttype   = row[1]
        bg = MEAN_BG if is_mean else (LEG_BG if ttype == "Legible" else AMB_BG)
        # draw each cell
        for ci in range(n_cols):
            ax.add_patch(plt.Rectangle((xs[ci], y), col_widths[ci], ROW_H,
                                       fc=bg, ec=GRID_COLOR, lw=0.5, zorder=1))
        for ci, val in enumerate(row):
            fw = "bold"
            ax.text(cx[ci], y + ROW_H / 2, val,
                    ha="center", va="center", fontsize=11,
                    color=TEXT_DARK, fontweight=fw, zorder=3)

    # outer border
    border_h = HEAD_H + n_rows * ROW_H
    ax.add_patch(plt.Rectangle((0, head_y - n_rows * ROW_H), total_w, border_h,
                                fc="none", ec="#374151", lw=1.5, zorder=5))

    # footer
    footer_y = head_y - n_rows * ROW_H - 0.25
    ax.text(0.05, footer_y, footer_text,
            fontsize=9, color="#4B5563", va="top")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10, color="#111827")


def fig_table_best(
    human_acc: dict,
    providers: dict,
    out_stem: Path,
) -> None:
    """Clean summary table: best 3 VLMs + Human, accuracy only (C=wrong)."""
    leg_vids = [v for v, t in TRAJ_TYPE.items() if t == "Legible"]
    amb_vids = [v for v, t in TRAJ_TYPE.items() if t == "Ambiguous"]
    all_vids  = sorted(leg_vids) + sorted(amb_vids)

    entity_keys = ["Human (Part 1)"] + [n for n in BEST_VLM_LABELS if n in providers]
    col_headers = ["Video", "Type"]
    for ek in entity_keys:
        col_headers.append(ek.replace("Human (Part 1)", "Reference"))

    rows_data = []
    for vid in all_vids:
        ttype = TRAJ_TYPE[vid]
        label = VID_LABEL[vid].split(": ")[1]
        row = [label, ttype]
        for ek in entity_keys:
            val = (human_acc["per_video"].get(vid, 0)
                   if ek == "Human (Part 1)"
                   else providers[ek]["accuracy"]["per_video"].get(vid, 0))
            row.append(f"{val:.0f}%")
        rows_data.append(row)

    mean_row = ["Mean", "—"]
    for ek in entity_keys:
        val = (human_acc["overall"]
               if ek == "Human (Part 1)"
               else providers[ek]["accuracy"]["overall"])
        mean_row.append(f"{val:.1f}%")
    rows_data.append(mean_row)

    n_rows = len(rows_data)
    ROW_H  = 0.56
    HEAD_H = 0.64

    col_widths = [1.7, 1.2] + [1.5] * len(entity_keys)
    fig_w = sum(col_widths) + 0.5
    fig_h = HEAD_H + n_rows * ROW_H + 0.65

    # header bg: neutral dark for Video/Type, brand colors for entities
    BRAND_COLORS = {
        "Reference":       "#1B4332",
        "Gemini 3 Pro":   "#1D4ED8",
        "GPT-5.4":        "#991B1B",
        "Claude Opus 4.5":"#78350F",
        "Claude Opus 4-6":"#92400E",
        "GPT-4o":         "#B91C1C",
    }
    header_bgs = ["#374151", "#374151"] + [
        BRAND_COLORS.get(ek.replace("Human (Part 1)", "Reference"), "#374151")
        for ek in entity_keys
    ]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_clean_table(
        ax, fig_h, col_headers, col_widths, rows_data, header_bgs,
        title="Reference vs VLM — Goal Inference Accuracy",
        footer_text="C (uncertain) counted as wrong  ·  Blue rows = Legible  ·  Red rows = Ambiguous",
    )
    fig.tight_layout()
    save_fig(fig, out_stem)
    print(f"  Saved: {out_stem}.*")


def fig_table_agreement(
    providers: dict,
    out_stem: Path,
) -> None:
    """Clean summary table: human–VLM agreement per video, best models."""
    leg_vids = [v for v, t in TRAJ_TYPE.items() if t == "Legible"]
    amb_vids = [v for v, t in TRAJ_TYPE.items() if t == "Ambiguous"]
    all_vids  = sorted(leg_vids) + sorted(amb_vids)

    entity_keys = [n for n in BEST_VLM_LABELS if n in providers]
    col_headers = ["Video", "Type"] + entity_keys

    rows_data = []
    for vid in all_vids:
        ttype = TRAJ_TYPE[vid]
        label = VID_LABEL[vid].split(": ")[1]
        row = [label, ttype]
        for ek in entity_keys:
            val = providers[ek]["agreement"]["per_video"].get(vid, 0)
            row.append(f"{val:.0f}%")
        rows_data.append(row)

    mean_row = ["Mean", "—"]
    for ek in entity_keys:
        mean_row.append(f"{providers[ek]['agreement']['overall']:.1f}%")
    rows_data.append(mean_row)

    n_rows = len(rows_data)
    ROW_H  = 0.56
    HEAD_H = 0.64

    col_widths = [1.7, 1.2] + [1.65] * len(entity_keys)
    fig_w = sum(col_widths) + 0.5
    fig_h = HEAD_H + n_rows * ROW_H + 0.65

    BRAND_COLORS = {
        "Gemini 3 Pro":   "#1D4ED8",
        "GPT-5.4":        "#991B1B",
        "Claude Opus 4.5":"#78350F",
        "Claude Opus 4-6":"#92400E",
        "GPT-4o":         "#B91C1C",
    }
    header_bgs = ["#374151", "#374151"] + [
        BRAND_COLORS.get(ek, "#374151") for ek in entity_keys
    ]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_clean_table(
        ax, fig_h, col_headers, col_widths, rows_data, header_bgs,
        title="Human vs VLM — Choice Agreement",
        footer_text="Agreement: % of human Part-1 choices matched by VLM at the same timepoint  ·  Blue = Legible  ·  Red = Ambiguous",
    )
    fig.tight_layout()
    save_fig(fig, out_stem)
    print(f"  Saved: {out_stem}.*")


def fig_accuracy_bar(human_acc: dict, providers: dict, out_stem: Path) -> None:
    """Vertical grouped bar chart: per-video accuracy, best models + human."""
    from matplotlib.patches import Patch

    leg_vids = sorted(v for v, t in TRAJ_TYPE.items() if t == "Legible")
    amb_vids = sorted(v for v, t in TRAJ_TYPE.items() if t == "Ambiguous")
    all_vids = leg_vids + amb_vids

    # Best models only
    names_all = ["Human (Part 1)"] + [n for n in BEST_VLM_LABELS if n in providers]
    colors    = [PROVIDER_COLORS[n] for n in names_all]
    n_models  = len(names_all)

    x     = np.arange(len(all_vids))
    width = 0.17
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (name, color) in enumerate(zip(names_all, colors)):
        vals = [human_acc["per_video"].get(v, 0)
                if name == "Human (Part 1)"
                else providers[name]["accuracy"]["per_video"].get(v, 0)
                for v in all_vids]
        bars = ax.bar(x + offsets[i], vals, width * 0.9, label=name,
                      color=color, alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        f"{int(val)}", ha="center", va="bottom", fontsize=7,
                        color=color, fontweight="bold")

    # Chance line
    ax.axhline(50, color="grey", lw=1.0, ls=":", alpha=0.55, zorder=1)
    ax.text(len(all_vids) - 0.45, 51.5, "Chance (50%)", fontsize=8,
            color="grey", style="italic", ha="right")

    # Section shading
    n_leg = len(leg_vids)
    ax.axvspan(-0.5, n_leg - 0.5,               alpha=0.06, color="#1B6CA8", zorder=0)
    ax.axvspan(n_leg - 0.5, len(all_vids) - 0.5, alpha=0.06, color="#C0392B", zorder=0)
    ax.text(n_leg / 2 - 0.5,        103, "Legible",   ha="center", fontsize=10,
            color="#1B6CA8", fontweight="bold")
    ax.text(n_leg + len(amb_vids) / 2 - 0.5, 103, "Ambiguous", ha="center", fontsize=10,
            color="#C0392B", fontweight="bold")

    vid_labels = [VID_LABEL[v].split(": ")[1] for v in all_vids]
    ax.set_xticks(x)
    ax.set_xticklabels(vid_labels, fontsize=10)
    ax.set_ylabel("Accuracy  (%;  C = uncertain counted as wrong)", fontsize=11)
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True, ls="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_labels = ["Reference (Part 1)" if n == "Human (Part 1)" else n for n in names_all]
    handles = [Patch(facecolor=c, label=lbl, alpha=0.88) for lbl, c in zip(legend_labels, colors)]
    ax.legend(handles=handles, fontsize=9.5, loc="upper right", ncol=2,
              framealpha=0.93, edgecolor="#cccccc")

    ax.set_title("Goal Inference Accuracy per Video\n"
                 "Reference (Part 1) vs Best VLMs  ·  31 shared timepoints",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout(pad=1.4)
    save_fig(fig, out_stem)
    print(f"  Saved: {out_stem}.*")


def fig_agreement_bar(providers: dict, out_stem: Path) -> None:
    """Vertical grouped bar chart: per-video human–VLM agreement, best models."""
    from matplotlib.patches import Patch

    leg_vids = sorted(v for v, t in TRAJ_TYPE.items() if t == "Legible")
    amb_vids = sorted(v for v, t in TRAJ_TYPE.items() if t == "Ambiguous")
    all_vids = leg_vids + amb_vids

    # Best models only
    prov_names = [n for n in BEST_VLM_LABELS if n in providers]
    colors     = [PROVIDER_COLORS[n] for n in prov_names]
    n_models   = len(prov_names)

    x     = np.arange(len(all_vids))
    width = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (name, color) in enumerate(zip(prov_names, colors)):
        vals = [providers[name]["agreement"]["per_video"].get(v, 0) for v in all_vids]
        bars = ax.bar(x + offsets[i], vals, width * 0.9, label=name,
                      color=color, alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        f"{int(val)}", ha="center", va="bottom", fontsize=7.5,
                        color=color, fontweight="bold")

    # Section shading
    n_leg = len(leg_vids)
    ax.axvspan(-0.5, n_leg - 0.5,                alpha=0.06, color="#1B6CA8", zorder=0)
    ax.axvspan(n_leg - 0.5, len(all_vids) - 0.5, alpha=0.06, color="#C0392B", zorder=0)
    ax.text(n_leg / 2 - 0.5,                   103, "Legible",   ha="center", fontsize=10,
            color="#1B6CA8", fontweight="bold")
    ax.text(n_leg + len(amb_vids) / 2 - 0.5,   103, "Ambiguous", ha="center", fontsize=10,
            color="#C0392B", fontweight="bold")

    vid_labels = [VID_LABEL[v].split(": ")[1] for v in all_vids]
    ax.set_xticks(x)
    ax.set_xticklabels(vid_labels, fontsize=10)
    ax.set_ylabel("Human–VLM Agreement  (%)", fontsize=11)
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True, ls="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [Patch(facecolor=c, label=n, alpha=0.88) for n, c in zip(prov_names, colors)]
    ax.legend(handles=handles, fontsize=9.5, loc="lower right", ncol=1,
              framealpha=0.93, edgecolor="#cccccc")

    ax.set_title("Human–VLM Choice Agreement per Video\n"
                 "% of Part-1 human choices matched by VLM at the same timepoint",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout(pad=1.4)
    save_fig(fig, out_stem)
    print(f"  Saved: {out_stem}.*")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini",  default="analysis_3/results_gemini_3_pro_preview.jsonl")
    parser.add_argument("--gemini2", default="analysis_3/results_gemini_2_5_flash.jsonl",
                        help="Second Gemini model for comparison")
    parser.add_argument("--openai",  default="outputs/results_gpt54.jsonl")
    parser.add_argument("--openai2", default="outputs/results_gpt4o.jsonl",
                        help="Optional second OpenAI model for comparison")
    parser.add_argument("--claude",  default="outputs/results_claude_opus46.jsonl")
    parser.add_argument("--claude2", default="outputs/results_claude_opus.jsonl",
                        help="Optional second Claude model for comparison")
    parser.add_argument("--out",     default="outputs/part1_comparison")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load human Part-1 data ----
    human_df = load_human_part1()
    if human_df.empty:
        print("ERROR: No human Part-1 data found. Check PARTICIPANT_DATA_DIR.")
        sys.exit(1)

    human_acc       = human_accuracy_per_video(human_df)
    human_comm_acc  = human_committed_accuracy_per_video(human_df)

    print(f"\n{'='*60}")
    print(f"Human Part-1 Accuracy  (C counts as WRONG - matches original study)")
    print(f"  Overall: {human_acc['overall']:.2f}%")
    for v in sorted(GROUND_TRUTH):
        print(f"  {v:<30} {human_acc['per_video'].get(v, 0):.1f}%")

    pct_committed = human_comm_acc['n_committed'] / human_comm_acc['n_total'] * 100
    print(f"\nHuman Part-1 Committed Accuracy  (C EXCLUDED from denominator)")
    print(f"  Overall: {human_comm_acc['overall']:.2f}%  "
          f"({human_comm_acc['n_committed']}/{human_comm_acc['n_total']} choices committed = {pct_committed:.0f}%)")
    for v in sorted(GROUND_TRUTH):
        print(f"  {v:<30} {human_comm_acc['per_video'].get(v, 0):.1f}%")
    print(f"{'='*60}")

    # ---- Load VLM files ----
    raw_vlm = {}
    model_file_map = [
        ("Gemini 2.5 Flash", args.gemini2),
        ("Gemini 3 Pro",     args.gemini),
        ("GPT-4o",           args.openai2),
        ("GPT-5.4",          args.openai),
        ("Claude Opus 4.5",  args.claude2),
        ("Claude Opus 4-6",  args.claude),
    ]
    for label, path in model_file_map:
        p = Path(path)
        if not p.exists():
            print(f"  (skipping {label}: {path} not found)")
            continue
        df = load_jsonl(str(p))
        raw_vlm[label] = df
        print(f"  {label}: {len(df)} predictions loaded from {path}")

    if not raw_vlm:
        print("No VLM files found.")
        sys.exit(1)

    # ---- Align to shared (video_id, t_sec) across all three VLMs ----
    shared_keys = None
    for df in raw_vlm.values():
        keys = set(zip(df["video_id"], df["t_sec"].astype(int)))
        shared_keys = keys if shared_keys is None else shared_keys & keys

    # Also restrict to timepoints that appear in the human data
    human_keys = set(zip(human_df["video_id"], human_df["timepoint"]))
    shared_keys = shared_keys & human_keys

    print(f"\nAligned to {len(shared_keys)} shared (video_id, t_sec) timepoints "
          f"(intersection of all VLMs ∩ human Part-1 timepoints)")

    # ---- Compute per-provider metrics ----
    providers = {}
    for label, df in raw_vlm.items():
        acc  = vlm_accuracy_per_video(df, shared_keys)
        agr  = agreement_per_video(human_df, df, shared_keys)
        providers[label] = {"accuracy": acc, "agreement": agr}
        ab_rate = acc["n_abstain"] / acc["n_total"] * 100 if acc["n_total"] > 0 else 0
        comm_pct = acc["n_committed"] / acc["n_total"] * 100 if acc["n_total"] > 0 else 0
        print(f"\n{label}:")
        print(f"  Overall accuracy   (C=wrong)   : {acc['overall']:.2f}%  "
              f"({acc['n_abstain']}/{acc['n_total']} abstained = {ab_rate:.0f}%)")
        print(f"  Committed accuracy (C excluded) : {acc['committed_overall']:.2f}%  "
              f"({acc['n_committed']}/{acc['n_total']} committed = {comm_pct:.0f}%)")
        print(f"  Overall agreement : {agr['overall']:.2f}%")
        for v in sorted(GROUND_TRUTH):
            print(f"    {v:<30} acc={acc['per_video'].get(v,0):.1f}%  "
                  f"comm_acc={acc['committed_per_video'].get(v,0):.1f}%  "
                  f"agr={agr['per_video'].get(v,0):.1f}%")

    # ---- Save CSV ----
    rows_csv = []
    for v in sorted(GROUND_TRUTH):
        row = {
            "video_id":              v,
            "traj_type":             TRAJ_TYPE[v],
            "human_acc_pct":         round(human_acc["per_video"].get(v, 0), 1),
            "human_comm_acc_pct":    round(human_comm_acc["per_video"].get(v, 0), 1),
        }
        for name in providers:
            tag = name.replace(" ", "_").lower()
            row[f"{tag}_acc_pct"]      = round(providers[name]["accuracy"]["per_video"].get(v, 0), 1)
            row[f"{tag}_comm_acc_pct"] = round(providers[name]["accuracy"]["committed_per_video"].get(v, 0), 1)
            row[f"{tag}_agr_pct"]      = round(providers[name]["agreement"]["per_video"].get(v, 0), 1)
        rows_csv.append(row)
    # Mean row
    mean_row = {
        "video_id":           "MEAN",
        "traj_type":          "—",
        "human_acc_pct":      round(human_acc["overall"], 1),
        "human_comm_acc_pct": round(human_comm_acc["overall"], 1),
    }
    for name in providers:
        tag = name.replace(" ", "_").lower()
        mean_row[f"{tag}_acc_pct"]      = round(providers[name]["accuracy"]["overall"], 1)
        mean_row[f"{tag}_comm_acc_pct"] = round(providers[name]["accuracy"]["committed_overall"], 1)
        mean_row[f"{tag}_agr_pct"]      = round(providers[name]["agreement"]["overall"], 1)
    rows_csv.append(mean_row)

    csv_path = out_dir / "part1_comparison.csv"
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)
    print(f"\nSaved CSV → {csv_path}")

    # ---- Figures ----
    fig_human_choice_breakdown(human_df, out_dir / "figure_human_choice_breakdown")
    fig_summary_table(human_acc, providers, out_dir / "figure_summary_table")
    fig_summary_table_committed(human_comm_acc, providers, out_dir / "figure_summary_table_committed")
    fig_accuracy_bar(human_acc,  providers, out_dir / "figure_accuracy_bar")
    fig_agreement_bar(            providers, out_dir / "figure_agreement_bar")
    fig_table_best(human_acc, providers, out_dir / "figure_table_best")
    fig_table_agreement(providers, out_dir / "figure_table_agreement")

    print("\nDone.")


if __name__ == "__main__":
    main()
