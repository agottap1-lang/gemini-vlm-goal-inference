import json
from collections import Counter

rows = [json.loads(l) for l in open("outputs/results_gpt4o.jsonl", encoding="utf-8") if l.strip()]
print("Model:", rows[0]["model"])
print("Mode:", rows[0]["evaluation_mode"])
print("Total rows:", len(rows))
print("Choices overall:", dict(Counter(r["choice"] for r in rows)))
print()
for v in sorted(set(r["video_id"] for r in rows)):
    vr = [r for r in rows if r["video_id"] == v]
    gt = vr[0]["goal_gt"]
    cc = Counter(r["choice"] for r in vr)
    correct = sum(1 for r in vr if r["choice"] == gt)
    pAs = [r["pA"] for r in vr]
    pBs = [r["pB"] for r in vr]
    print(f"  {v}  GT={gt}  correct={correct}/{len(vr)}")
    print(f"    choices: A={cc.get('A',0)} B={cc.get('B',0)} C={cc.get('C',0)}")
    print(f"    mean pA={sum(pAs)/len(pAs):.2f}  mean pB={sum(pBs)/len(pBs):.2f}")
    # show each timepoint
    for r in sorted(vr, key=lambda x: x["t_sec"]):
        mark = "OK" if r["choice"] == gt else "WRONG" if r["choice"] != "C" else "C"
        print(f"      t={r['t_sec']:2d}s  choice={r['choice']}  pA={r['pA']:.2f} pB={r['pB']:.2f}  [{mark}]")
