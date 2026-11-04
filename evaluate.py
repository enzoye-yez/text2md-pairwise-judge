"""Coverage / precision of judge output against labels. No API calls.
labels jsonl: {"pid", "label"} with label "x" | "y" | "similar" (e.g. higher PRISM; |PRISM gap| <= 0.25 -> similar).
  coverage   = share of labelled (non-similar) pairs with |margin| >= t
  precision  = share of those where sign(margin) agrees with the label
  false_call = share of "similar" pairs with |margin| >= t (a confident call where the labels see no difference)
  acc_all    = sign agreement over every labelled pair (margin 0 counts 1/2)
usage: python evaluate.py out.jsonl labels.jsonl [--thresholds 0.6,0.7,0.8]
"""
import argparse, json

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("labels"); ap.add_argument("--thresholds", default="0.6,0.7,0.8")
a = ap.parse_args()
M = {d["pid"]: d["margin"] for d in map(json.loads, open(a.out)) if "error" not in d}
L = {d["pid"]: d["label"] for d in map(json.loads, open(a.labels)) if d["pid"] in M}
sg = {"x": -1, "y": 1}; lab = [p for p in L if L[p] != "similar"]; sim = [p for p in L if L[p] == "similar"]
ok = lambda p: (M[p] > 0) == (sg[L[p]] > 0)
print(f"{len(L)} pairs ({len(lab)} labelled, {len(sim)} similar)")
for t in map(float, a.thresholds.split(",")):
    c = [p for p in lab if abs(M[p]) >= t]
    prec = f"{sum(map(ok, c)) / len(c):.1%}" if c else "—"
    fc = f"{sum(abs(M[p]) >= t for p in sim) / len(sim):.1%}" if sim else "—"
    print(f"|margin| >= {t}: coverage {len(c) / max(len(lab), 1):.1%}  precision {prec} (n {len(c)})  false_call {fc}")
if lab:
    print(f"acc_all {sum(.5 if M[p] == 0 else float(ok(p)) for p in lab) / len(lab):.1%}")
