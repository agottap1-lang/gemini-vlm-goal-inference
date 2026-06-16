#!/usr/bin/env python3
"""Verify user_study_timepoints.csv metadata against actual VLM outputs."""

import json
from pathlib import Path

videos = {
    'amb_d_drawer_close': {'first_uncertain': 0, 'first_high_conf': 5, 'max_flip': 9, 'final': 19},
    'amb_l_block': {'first_uncertain': 0, 'first_high_conf': 6, 'max_flip': 8, 'final': 12},
    'amb_r_block': {'first_uncertain': 0, 'first_high_conf': 1, 'max_flip': 7, 'final': 14},
    'amb_to_drawer_close': {'first_uncertain': 0, 'first_high_conf': 2, 'max_flip': 10, 'final': 15},
    'le_d_drawer_close': {'first_uncertain': 0, 'first_high_conf': 3, 'max_flip': 8, 'final': 11},
    'le_l_block': {'first_uncertain': 0, 'first_high_conf': 4, 'max_flip': 4, 'final': 14},
    'le_r_block': {'first_uncertain': 0, 'first_high_conf': 4, 'max_flip': 5, 'final': 11},
    'le_t_drawer_close': {'first_uncertain': 0, 'first_high_conf': 1, 'max_flip': 5, 'final': 8}
}

print('=' * 90)
print('VERIFYING user_study_timepoints.csv AGAINST LATEST VLM RUNS')
print('=' * 90)
print()

all_match = True

for video_id, expected in videos.items():
    # Find latest run
    video_dir = Path('outputs') / video_id
    run_dirs = sorted(video_dir.glob('run_*_prefix'))
    if not run_dirs:
        print(f'{video_id}: NO RUNS FOUND ❌')
        all_match = False
        continue
    
    latest_file = run_dirs[-1] / 'results.jsonl'
    
    # Load predictions
    preds = []
    with open(latest_file) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    
    preds = sorted(preds, key=lambda x: x['t_sec'])
    
    print(f'=== {video_id} (using {run_dirs[-1].name}) ===')
    
    video_matches = True
    
    # Check first uncertain (choice=C at t=0)
    first = preds[0]
    first_unc_ok = first['choice'] == 'C' and first['t_sec'] == 0
    first_unc_match = '✅' if first_unc_ok else '❌'
    print(f'  first_uncertain: Expected={expected["first_uncertain"]}, Actual=t={first["t_sec"]} choice={first["choice"]} {first_unc_match}')
    if not first_unc_ok:
        video_matches = False
    
    # Check first high confidence (>=90)
    high_conf = next((p for p in preds if p['confidence'] >= 90), None)
    if high_conf:
        hc_ok = high_conf['t_sec'] == expected['first_high_conf']
        hc_match = '✅' if hc_ok else '❌'
        print(f'  first_high_conf: Expected={expected["first_high_conf"]}, Actual=t={high_conf["t_sec"]} (conf={high_conf["confidence"]}, choice={high_conf["choice"]}) {hc_match}')
        if not hc_ok:
            video_matches = False
    else:
        print(f'  first_high_conf: Expected={expected["first_high_conf"]}, Actual=NONE ❌')
        video_matches = False
    
    # Check final
    last = preds[-1]
    final_ok = last['t_sec'] == expected['final']
    final_match = '✅' if final_ok else '❌'
    print(f'  final:           Expected={expected["final"]}, Actual={last["t_sec"]} {final_match}')
    if not final_ok:
        video_matches = False
    
    # Check flips
    choices = [p['choice'] for p in preds]
    flips = [preds[i]['t_sec'] for i in range(1, len(preds)) if choices[i] != choices[i-1]]
    max_flip_actual = flips[-1] if flips else None
    flip_ok = max_flip_actual == expected['max_flip']
    flip_match = '✅' if flip_ok else '⚠️'
    flip_str = ','.join(map(str, flips)) if flips else 'NONE'
    print(f'  max_flip:        Expected={expected["max_flip"]}, Actual={max_flip_actual} (all flips: {flip_str}) {flip_match}')
    if not flip_ok:
        video_matches = False
    
    if not video_matches:
        all_match = False
    
    print()

print('=' * 90)
if all_match:
    print('✅ VERDICT: All metadata in user_study_timepoints.csv MATCHES latest VLM outputs!')
else:
    print('❌ VERDICT: Some metadata does NOT match - CSV needs updating!')
print('=' * 90)
