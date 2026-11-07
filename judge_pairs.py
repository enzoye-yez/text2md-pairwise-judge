"""Judge md pairs with terra (prompt v1b by default, --prompt v1b.1 for the dpo_pick version), one call per pair.
Input jsonl, one pair per line: {"pid", "prompt", "aspect_ratio"?, "md_x", "md_y"} (pid must be unique; it fixes the A/B order).
Output jsonl: {"pid", "order", "probs", "margin", "reason", "usage", "model"}; margin > 0 means md_y is better. Resumable.
usage: OPENAI_API_KEY=... python judge_pairs.py pairs.jsonl out.jsonl [--concurrency 24] [--threshold 0.8] [--prompt v1b|v1b.1]
"""
import argparse, asyncio, json
import pairwise_judge as pj

ap = argparse.ArgumentParser(); ap.add_argument("pairs"); ap.add_argument("out")
ap.add_argument("--concurrency", type=int, default=24); ap.add_argument("--threshold", type=float, default=0.8)
ap.add_argument("--prompt", default="v1b", choices=["v1b", "v1b.1"])
a = ap.parse_args()
pairs = [json.loads(l) for l in open(a.pairs) if l.strip()]
asyncio.run(pj.run(pairs, a.out, a.concurrency, a.prompt))
want = {p["pid"] for p in pairs}
rows = [d for d in map(json.loads, open(a.out)) if "error" not in d and d["pid"] in want]
kept = [d for d in rows if abs(d["margin"]) >= a.threshold]
c = pj.cost(rows)
print(f"{len(rows)}/{len(pairs)} pairs judged; |margin| >= {a.threshold}: {len(kept)} ({len(kept) / max(len(rows), 1):.1%}), "
      f"{sum(d['margin'] > 0 for d in kept)} pick y, {sum(d['margin'] < 0 for d in kept)} pick x")
print(f"tokens: input {c['input']:,} (cached {c['cached']:,}), output {c['output']:,}; ≈ ${c['usd']:.2f} (${c['usd'] / max(c['calls'], 1):.4f}/pair)")
