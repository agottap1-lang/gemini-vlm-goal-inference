#!/usr/bin/env python3
"""Explain max_flip logic and why two values are wrong."""

import json
from pathlib import Path

print("=" * 90)
print("EXPLAINING max_flip LOGIC")
print("=" * 90)
print()
print("max_flip Definition:")
print("  The LAST timepoint where the VLM's choice CHANGED compared to previous timepoint")
print()
print("Algorithm:")
print("  1. Go through timepoints in order: t=0,1,2,3,4,5...")
print("  2. Compare choice at each t with choice at t-1")
print("  3. If they differ, that's a FLIP")
print("  4. max_flip = the timepoint of the FINAL/LAST flip")
print()
print("=" * 90)
print()

# Check amb_d_drawer_close
print("=== amb_d_drawer_close ===")
print("CSV says: max_flip=9")
print()

video_dir = Path('outputs/amb_d_drawer_close')
latest = sorted(video_dir.glob('run_*_prefix'))[-1]
file = latest / 'results.jsonl'

preds = []
with open(file) as f:
    for line in f:
        if line.strip():
            preds.append(json.loads(line))

preds = sorted(preds, key=lambda x: x['t_sec'])

print("Full choice sequence:")
for i, p in enumerate(preds):
    print(f"  t={p['t_sec']:2d}: choice={p['choice']} (confidence={p['confidence']:3d})", end="")
    if i > 0 and preds[i]['choice'] != preds[i-1]['choice']:
        print(f" ← FLIP from {preds[i-1]['choice']}")
    else:
        print()

print()
choices = [p['choice'] for p in preds]
flips = [preds[i]['t_sec'] for i in range(1, len(preds)) if choices[i] != choices[i-1]]
print(f"Flips occur at timepoints: {flips}")
print(f"LAST flip (max_flip): {flips[-1]}")
print(f"CSV value: 9 ❌ (should be {flips[-1]})")
print()
print("Why wrong? The choice kept changing after t=9. At t=10 and t=11, it flipped again.")
print()

# Check amb_r_block
print("=" * 90)
print()
print("=== amb_r_block ===")
print("CSV says: max_flip=7")
print()

video_dir = Path('outputs/amb_r_block')
latest = sorted(video_dir.glob('run_*_prefix'))[-1]
file = latest / 'results.jsonl'

preds = []
with open(file) as f:
    for line in f:
        if line.strip():
            preds.append(json.loads(line))

preds = sorted(preds, key=lambda x: x['t_sec'])

print("Full choice sequence:")
for i, p in enumerate(preds):
    print(f"  t={p['t_sec']:2d}: choice={p['choice']} (confidence={p['confidence']:3d})", end="")
    if i > 0 and preds[i]['choice'] != preds[i-1]['choice']:
        print(f" ← FLIP from {preds[i-1]['choice']}")
    else:
        print()

print()
choices = [p['choice'] for p in preds]
flips = [preds[i]['t_sec'] for i in range(1, len(preds)) if choices[i] != choices[i-1]]
print(f"Flips occur at timepoints: {flips}")
print(f"LAST flip (max_flip): {flips[-1]}")
print(f"CSV value: 7 ❌ (should be {flips[-1]})")
print()
print("Why wrong? The choice changed at t=7, but then changed AGAIN at t=10.")
print("         The LAST flip is at t=10, not t=7.")
print()

print("=" * 90)
print()
print("✅ CONCLUSION:")
print("   max_flip = Last timepoint where prediction choice changed")
print()
print("   amb_d_drawer_close: Update from 9 → 11")
print("   amb_r_block:        Update from 7 → 10")
print()
print("=" * 90)
