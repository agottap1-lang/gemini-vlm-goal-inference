"""
Embeds pre-computed outputs (figures + text) into the notebook cells.
After running this, the .ipynb opens with all outputs already rendered.
"""
import json, base64, glob
import numpy as np
import pandas as pd
from pathlib import Path

# ── Load the notebook ───────────────────────────────────────────────────────
nb_path = Path("vlm_legibility_analysis.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

def png_output(png_path):
    """Return a Jupyter display_data output dict with a PNG image."""
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "output_type": "display_data",
        "data": {"image/png": b64, "text/plain": ["<Figure>"]},
        "metadata": {"image/png": {"width": 1200}}
    }

def text_output(text):
    """Return a stream stdout output dict."""
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": text if isinstance(text, list) else [text]
    }

def df_html_output(df, title=""):
    """Return a display_data output with an HTML table."""
    html = df.to_html(classes="dataframe", border=0, float_format=lambda x: f"{x:.1f}")
    return {
        "output_type": "display_data",
        "data": {
            "text/html": [html],
            "text/plain": [df.to_string()]
        },
        "metadata": {}
    }

# ── Recompute all analysis outputs ─────────────────────────────────────────
import json as _json, glob as _glob, warnings
warnings.filterwarnings("ignore")

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rows.append(_json.loads(line))
            except: pass
    return pd.DataFrame(rows)

BASE = Path("outputs")
VIDEO_IDS = ["amb_l_block","amb_r_block","amb_d_drawer_close","amb_to_drawer_close",
             "le_l_block","le_r_block","le_d_drawer_close","le_t_drawer_close"]
MODEL_LABELS = {"gemini-2.5-flash":"Gemini 2.5 Flash","gpt-4o":"GPT-4o",
                "gpt-5.4":"GPT-5.4","claude-opus-4-5":"Claude Opus 4.5"}
HUMAN_ACCURACY = 0.613

gemini_dfs = []
for vid in VIDEO_IDS:
    runs = sorted(_glob.glob(str(BASE / vid / "run_*_prefix" / "results.jsonl")))
    if runs: gemini_dfs.append(load_jsonl(runs[0]))
gemini_df = pd.concat(gemini_dfs, ignore_index=True)

other_dfs = []
load_log = [f"Gemini rows: {len(gemini_df)}\n"]
for fname in ["results_gpt4o.jsonl","results_gpt54.jsonl","results_claude_opus.jsonl"]:
    p = BASE / fname
    if p.exists():
        df = load_jsonl(p)
        other_dfs.append(df)
        load_log.append(f"Loaded {fname}: {len(df)} rows | model={df['model'].iloc[0]}\n")

all_df = pd.concat([gemini_df] + other_dfs, ignore_index=True)
all_df["model_label"] = all_df["model"].map(MODEL_LABELS).fillna(all_df["model"])
all_df["traj_type"] = all_df["video_id"].apply(lambda x: "Ambiguous" if x.startswith("amb") else "Legible")
all_df["correct"] = (all_df["choice"] == all_df["goal_gt"]) & (all_df["choice"] != "C")
load_log.append(f"\nTotal rows: {len(all_df)} | Videos: {all_df['video_id'].nunique()} | Models: {all_df['model_label'].nunique()}\n")

acc_df = all_df.groupby(["video_id","traj_type","model_label"]).agg(accuracy=("correct","mean")).reset_index()

mean_acc_lines = ["=== Mean Goal Inference Accuracy ===\n"]
for m, g in sorted(all_df.groupby("model_label"), key=lambda x: -x[1]["correct"].mean()):
    mean_acc_lines.append(f"  {m:<22} {g['correct'].mean()*100:.1f}%\n")
mean_acc_lines.append(f"  {'Human (thesis study)':<22} {HUMAN_ACCURACY*100:.1f}%\n")

def ttl(group):
    lr = group[group["legible"] == "legible_now"]
    return lr["t_sec"].min() if len(lr) > 0 else np.nan
ttl_df = all_df.groupby(["video_id","traj_type","model_label"]).apply(ttl).reset_index(name="time_to_legibility")
ttl_sample = ttl_df.head(8).to_string(index=False)
metrics_out = mean_acc_lines + [f"\nSample time-to-legibility:\n{ttl_sample}\n"]

# Summary table
summary = all_df.groupby("model_label").agg(
    mean_accuracy=("correct","mean"),
    mean_confidence=("confidence","mean"),
    legible_rate=("legible", lambda x: (x=="legible_now").mean()),
    uncertain_rate=("choice", lambda x: (x=="C").mean()),
).reset_index()
summary[["mean_accuracy","mean_confidence","legible_rate","uncertain_rate"]] *= 100
summary = summary.rename(columns={"model_label":"Model","mean_accuracy":"Accuracy (%)","mean_confidence":"Avg Conf (%)","legible_rate":"Legible Rate (%)","uncertain_rate":"Uncertain (%)"})
summary = summary.sort_values("Accuracy (%)", ascending=False)
human_row = pd.DataFrame([{"Model":"Human (8 participants)","Accuracy (%)":61.3,"Avg Conf (%)":"—","Legible Rate (%)":"—","Uncertain (%)":"—"}])
summary = pd.concat([summary, human_row], ignore_index=True).set_index("Model")

# Head preview of data
head_preview = all_df[["video_id","traj_type","model_label","t_sec","choice","confidence","correct"]].head(8)

# ── Map cell IDs to their outputs ───────────────────────────────────────────
cell_outputs = {
    "b2c3d4e5": [text_output("Libraries loaded.\n")],
    "c3d4e5f6": [text_output(load_log), df_html_output(head_preview)],
    "d4e5f6g7": [text_output(metrics_out)],
    "e5f6g7h8": [text_output("Figure 1 saved.\n"), png_output("outputs/fig1_accuracy_by_traj_type.png")],
    "f6g7h8i9": [text_output("Figure 2 saved.\n"), png_output("outputs/fig2_legibility_timeline.png")],
    "g7h8i9j0": [text_output("Figure 3 saved.\n"), png_output("outputs/fig3_confidence_distribution.png")],
    "h8i9j0k1": [text_output("Figure 4 saved.\n"), png_output("outputs/fig4_time_to_legibility.png")],
    "i9j0k1l2": [text_output("Figure 5 saved.\n"), png_output("outputs/fig5_accuracy_heatmap.png"),
                 text_output("\nPer-video accuracy table:\n"),
                 df_html_output(acc_df.pivot_table(index="video_id",columns="model_label",values="accuracy").mul(100).round(1))],
    "j0k1l2m3": [text_output("=== Full Summary Table ===\n"), df_html_output(summary.reset_index())],
}

# ── Inject outputs into notebook cells ─────────────────────────────────────
count = 0
for cell in nb["cells"]:
    cid = cell.get("id", "")
    if cid in cell_outputs:
        cell["outputs"] = cell_outputs[cid]
        cell["execution_count"] = count + 1
        count += 1

# ── Save notebook ───────────────────────────────────────────────────────────
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Done. Embedded outputs into {count} cells -> {nb_path}")
