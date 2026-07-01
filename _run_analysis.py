import json
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 150, 'savefig.bbox': 'tight'})

MODEL_LABELS = {
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gpt-4o':           'GPT-4o',
    'gpt-5.4':          'GPT-5.4',
    'claude-opus-4-5':  'Claude Opus 4.5',
}
MODEL_COLORS = {
    'Gemini 2.5 Flash': '#4285F4',
    'GPT-4o':           '#10A37F',
    'GPT-5.4':          '#0a6644',
    'Claude Opus 4.5':  '#D4762B',
}
HUMAN_ACCURACY = 0.613
HUMAN_COLOR    = '#555555'


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(rows)


base = Path('outputs')
VIDEO_IDS = [
    'amb_l_block', 'amb_r_block', 'amb_d_drawer_close', 'amb_to_drawer_close',
    'le_l_block',  'le_r_block',  'le_d_drawer_close',  'le_t_drawer_close',
]

gemini_dfs = []
for vid in VIDEO_IDS:
    runs = sorted(glob.glob(str(base / vid / 'run_*_prefix' / 'results.jsonl')))
    if runs:
        gemini_dfs.append(load_jsonl(runs[0]))
gemini_df = pd.concat(gemini_dfs, ignore_index=True)

other_dfs = []
for fname in ['results_gpt4o.jsonl', 'results_gpt54.jsonl', 'results_claude_opus.jsonl']:
    p = base / fname
    if p.exists():
        df = load_jsonl(p)
        other_dfs.append(df)

all_df = pd.concat([gemini_df] + other_dfs, ignore_index=True)
all_df['model_label'] = all_df['model'].map(MODEL_LABELS).fillna(all_df['model'])
all_df['traj_type']   = all_df['video_id'].apply(
    lambda x: 'Ambiguous' if x.startswith('amb') else 'Legible'
)
all_df['correct'] = (all_df['choice'] == all_df['goal_gt']) & (all_df['choice'] != 'C')
print(f"Loaded {len(all_df)} rows, {all_df['model_label'].nunique()} models")

acc_df = (
    all_df
    .groupby(['video_id', 'traj_type', 'model_label'])
    .agg(accuracy=('correct', 'mean'))
    .reset_index()
)

# ── FIG 1 — Accuracy by Trajectory Type ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, ttype in zip(axes, ['Legible', 'Ambiguous']):
    subset   = acc_df[acc_df['traj_type'] == ttype]
    mean_pm  = (
        subset.groupby('model_label')['accuracy']
        .mean().reset_index()
        .sort_values('accuracy', ascending=False)
    )
    colors = [MODEL_COLORS.get(m, '#888') for m in mean_pm['model_label']]
    bars   = ax.bar(
        mean_pm['model_label'], mean_pm['accuracy'] * 100,
        color=colors, alpha=0.85, edgecolor='white', linewidth=1.2
    )
    ax.axhline(
        HUMAN_ACCURACY * 100, color=HUMAN_COLOR, linestyle='--',
        linewidth=1.8, label=f'Human baseline ({HUMAN_ACCURACY*100:.0f}%)'
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    ax.set_title(f'{ttype} Trajectories', fontsize=13, fontweight='bold')
    ax.set_ylabel('Goal Inference Accuracy (%)' if ax == axes[0] else '')
    ax.set_ylim(0, 108)
    ax.tick_params(axis='x', rotation=15)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

fig.suptitle(
    'VLMs match or exceed human accuracy on legible trajectories — ambiguous ones expose the gap',
    fontsize=12, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('outputs/fig1_accuracy_by_traj_type.png', dpi=150)
plt.close()
print('Fig 1 saved')

# ── FIG 2 — Legibility Timeline ────────────────────────────────────────────
SHOWCASE = [
    ('Legible — Left Block Pick',    'le_l_block'),
    ('Ambiguous — Left Block Pick',  'amb_l_block'),
    ('Legible — Drawer Close',       'le_d_drawer_close'),
    ('Ambiguous — Drawer Close',     'amb_d_drawer_close'),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
axes = axes.flatten()
for ax, (title, vid) in zip(axes, SHOWCASE):
    vid_df = all_df[all_df['video_id'] == vid]
    for ml, color in MODEL_COLORS.items():
        m_df = vid_df[vid_df['model_label'] == ml].sort_values('t_sec')
        if m_df.empty:
            continue
        ax.plot(
            m_df['t_sec'], m_df['confidence'],
            color=color, linewidth=2, marker='o', markersize=4,
            label=ml, alpha=0.85
        )
    ax.axhline(52, color='black', linestyle=':', linewidth=1.4,
               label='Decision threshold (52%)')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Confidence (%)' if ax in [axes[0], axes[2]] else '')
    ax.set_ylim(40, 108)
    ax.legend(fontsize=8, loc='lower right')
    sns.despine(ax=ax)

fig.suptitle(
    'Confidence over time — legible trajectories commit earlier; ambiguous ones show high variance',
    fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('outputs/fig2_legibility_timeline.png', dpi=150)
plt.close()
print('Fig 2 saved')

# ── FIG 3 — Confidence Distribution ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for ax, ml in zip(axes, list(MODEL_COLORS.keys())):
    m_df = all_df[all_df['model_label'] == ml]
    for leg_status, color, label in [
        ('legible_now',     '#2ecc71', 'Legible now'),
        ('not_legible_yet', '#e74c3c', 'Not legible yet'),
    ]:
        subset = m_df[m_df['legible'] == leg_status]['confidence']
        if not subset.empty:
            ax.hist(
                subset, bins=20, alpha=0.6, color=color,
                label=f'{label} (n={len(subset)})', edgecolor='white'
            )
    ax.axvline(52, color='black', linestyle='--', linewidth=1.4, label='Threshold (52%)')
    ax.set_title(ml, fontsize=11, fontweight='bold', color=MODEL_COLORS[ml])
    ax.set_xlabel('Confidence (%)')
    ax.set_ylabel('Frame Count')
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

fig.suptitle(
    'Confidence distribution — high confidence aligns with legibility, but degree varies by model',
    fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('outputs/fig3_confidence_distribution.png', dpi=150)
plt.close()
print('Fig 3 saved')

# ── FIG 4 — Time-to-Legibility ─────────────────────────────────────────────
def time_to_legibility(group):
    lr = group[group['legible'] == 'legible_now']
    return lr['t_sec'].min() if len(lr) > 0 else np.nan

ttl_df = (
    all_df
    .groupby(['video_id', 'traj_type', 'model_label'])
    .apply(time_to_legibility)
    .reset_index(name='time_to_legibility')
)
ttl_valid = ttl_df.dropna(subset=['time_to_legibility'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ttype in zip(axes, ['Legible', 'Ambiguous']):
    subset   = ttl_valid[ttl_valid['traj_type'] == ttype]
    mean_ttl = (
        subset.groupby('model_label')['time_to_legibility']
        .mean().reset_index()
        .sort_values('time_to_legibility')
    )
    colors = [MODEL_COLORS.get(m, '#888') for m in mean_ttl['model_label']]
    bars   = ax.barh(
        mean_ttl['model_label'], mean_ttl['time_to_legibility'],
        color=colors, alpha=0.85, edgecolor='white', linewidth=1.2
    )
    for bar in bars:
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.1f}s',
            va='center', fontsize=10, fontweight='bold'
        )
    ax.set_title(f'{ttype} Trajectories', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Time-to-Legibility (seconds)')
    ax.set_xlim(0, mean_ttl['time_to_legibility'].max() + 2)
    sns.despine(ax=ax)

fig.suptitle(
    'Time-to-legibility — earlier detection on legible; longer and variable on ambiguous',
    fontsize=12, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('outputs/fig4_time_to_legibility.png', dpi=150)
plt.close()
print('Fig 4 saved')

# ── FIG 5 — Per-Video Accuracy Heatmap ─────────────────────────────────────
pivot = acc_df.pivot_table(
    index='video_id', columns='model_label', values='accuracy'
) * 100
le_vids  = [v for v in VIDEO_IDS if v.startswith('le')]
amb_vids = [v for v in VIDEO_IDS if v.startswith('amb')]
pivot = pivot.loc[le_vids + amb_vids]

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    pivot, annot=True, fmt='.0f', cmap='RdYlGn',
    vmin=0, vmax=100, linewidths=0.5, linecolor='white',
    ax=ax, cbar_kws={'label': 'Accuracy (%)'}
)
ax.axhline(len(le_vids), color='black', linewidth=2)
ax.text(-0.5, len(le_vids) / 2, 'LEGIBLE', va='center',
        ha='right', fontsize=10, fontweight='bold', color='green', rotation=90)
ax.text(-0.5, len(le_vids) + len(amb_vids) / 2, 'AMBIGUOUS',
        va='center', ha='right', fontsize=10, fontweight='bold',
        color='#c0392b', rotation=90)
ax.set_title(
    'Per-video goal inference accuracy (%) — green = correct goal identified',
    fontsize=12, fontweight='bold', pad=12
)
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(axis='x', rotation=20)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig('outputs/fig5_accuracy_heatmap.png', dpi=150)
plt.close()
print('Fig 5 saved')

# ── Summary ─────────────────────────────────────────────────────────────────
print('\n=== FINAL ACCURACY SUMMARY ===')
for m, g in all_df.groupby('model_label'):
    print(f'  {m:<22} {g["correct"].mean()*100:.1f}%')
print(f'  {"Human (thesis)":<22} {HUMAN_ACCURACY*100:.1f}%')
print('\nAll charts saved to outputs/')
