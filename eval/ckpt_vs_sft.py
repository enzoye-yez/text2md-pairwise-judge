"""One checkpoint vs the SFT model on a fixed prompt set (PRISM 700 in our runs), md only, no rendering: terra v1b judges
every (SFT md, checkpoint md) pair in BOTH A/B orders (2 calls per prompt) and the two margins are averaged, so position
bias cancels. Appends one summary line to SUMMARY (tsv) and keeps every call in CK_DIR/judge_vs_sft.jsonl.

Inputs (jsonl, greedy mds; e.g. the results.jsonl of the text2md inference runtime):
  PROMPTS  {"id", "messages": [system, user]} where the user content is JSON {"prompt", "aspect_ratio"?}
  SFT_MDS / CK_MDS  {"id", "markdown", "format_valid", "finish_reason"}
Columns: win@0.6 / loss@0.6 = share of prompts where the averaged margin is >= 0.6 / <= -0.6 (checkpoint better / worse);
winrate = win + half of the rest; win(sign) / loss(sign) = share with margin > 0 / < 0; invalid / truncated counts; median words.
usage: python ckpt_vs_sft.py PROMPTS SFT_MDS CK_MDS CK_DIR NAME SUMMARY   (OPENAI_API_KEY in the environment)
"""
import asyncio, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pairwise_judge as pj  # noqa: E402

prompts_f, sft_f, ck_f, d, name, summary = sys.argv[1:7]
INP = {r["id"]: r for r in map(json.loads, open(prompts_f))}
SFT = {r["id"]: r["markdown"] for r in map(json.loads, open(sft_f))}
CK = {r["id"]: r for r in map(json.loads, open(ck_f))}
pairs = []
for c in sorted(INP):
    u = json.loads(INP[c]["messages"][1]["content"]); base = f"sft|ck|{c}"
    # second call: x/y swapped and a pid with the same A/B order, so the md shown as A changes between the two calls
    swap = next(f"{base}|swap{k}" for k in range(64) if pj.order_of(f"{base}|swap{k}") == pj.order_of(base))
    pairs.append({"pid": base, "prompt": u["prompt"], "aspect_ratio": u.get("aspect_ratio"), "md_x": SFT[c], "md_y": CK[c]["markdown"]})
    pairs.append({"pid": swap, "prompt": u["prompt"], "aspect_ratio": u.get("aspect_ratio"), "md_x": CK[c]["markdown"], "md_y": SFT[c]})
out = f"{d}/judge_vs_sft.jsonl"
asyncio.run(pj.run(pairs, out, 48))
by = {}
for r in map(json.loads, open(out)):
    if "error" not in r:
        c = r["pid"].split("|")[2]; by.setdefault(c, {})["swap" if "|swap" in r["pid"] else "base"] = r["margin"]
ms = [(o["base"] - o["swap"]) / 2 for o in by.values() if len(o) == 2]  # + = checkpoint better
win, loss = sum(m >= .6 for m in ms) / len(ms), sum(m <= -.6 for m in ms) / len(ms)
sw, sl = sum(m > 0 for m in ms) / len(ms), sum(m < 0 for m in ms) / len(ms)
rows = list(CK.values())
line = (f"{name}\t{len(ms)}\t{win:.3f}\t{loss:.3f}\t{win + .5 * (1 - win - loss):.3f}\t{sw:.3f}\t{sl:.3f}\t"
        f"{sum(not r['format_valid'] for r in rows)}\t{sum(r['finish_reason'] == 'length' for r in rows)}\t{statistics.median(len(r['markdown'].split()) for r in rows):.0f}")
if not os.path.exists(summary):
    open(summary, "w").write("ckpt\tn\twin@0.6\tloss@0.6\twinrate(ties half)\twin(sign)\tloss(sign)\tinvalid\ttruncated\tmedian_words\n")
open(summary, "a").write(line + "\n"); print(line)
