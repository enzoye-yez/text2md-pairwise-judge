"""PRISM-Bench scoring of rendered images with per-request alignment tracks (the recipe used for the GRPO checkpoints).

classify  one gpt-6.1-sol call per distinct prompt: assign one of the 7 PRISM tracks (prism_tracks.py) -> OUT/prism_track.jsonl
score     gpt-6.1-sol, reasoning low, native PNG with image detail "original", 3 rounds per image and dimension;
          alignment uses the template of the request's track ("## alignment · <track>" in prism_prompts_en.md),
          aesthetic the shared template -> OUT/prism_calls.jsonl (resumable: finished calls are skipped)
report    per group mean alignment / aesthetic / combined over the cases every group has (mean of 3 rounds,
          combined = mean of the two), by track, and paired differences between groups -> OUT/prism_report.json

Input jsonl, one image per line: {"id": unique, "group": e.g. checkpoint name, "case": shared across groups, "prompt", "image": PNG path}
Measured cost: ≈ $0.0054 per scoring call with prompt caching (6 calls per image), ≈ $0.002 per classification call.
usage: python prism_score.py classify|score|report IMAGES.jsonl OUT_DIR [--concurrency 24]   (OPENAI_API_KEY in the environment)
"""
import argparse, asyncio, base64, json, os, re
from collections import Counter, defaultdict
import httpx
import numpy as np
import prism_tracks as PT

HERE = os.path.dirname(os.path.abspath(__file__))
PRICE_IN, PRICE_CACHED, PRICE_OUT = 2.0, 0.2, 10.0  # gpt-6.1-sol, USD per 1M tokens (third-party listing)
ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["classify", "score", "report"]); ap.add_argument("images"); ap.add_argument("out")
ap.add_argument("--concurrency", type=int, default=24)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
ROWS = [json.loads(l) for l in open(a.images) if l.strip()]
TRACK_F, CALLS_F = f"{a.out}/prism_track.jsonl", f"{a.out}/prism_calls.jsonl"


def tracks():
    return {r["prompt"]: r["primary"] for r in map(json.loads, open(TRACK_F)) if "error" not in r} if os.path.exists(TRACK_F) else {}


def cost(us):
    c = lambda u: (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
    return sum((u["input_tokens"] - c(u)) * PRICE_IN + c(u) * PRICE_CACHED + u["output_tokens"] * PRICE_OUT for u in us) / 1e6


async def post_all(items, make_body, handle):
    sem = asyncio.Semaphore(a.concurrency)

    async def one(client, it):
        async with sem:
            body, err = make_body(it), None
            for att in range(6):
                try:
                    r = await client.post("https://api.openai.com/v1/responses", json=body)
                    if r.status_code in (429, 500, 502, 503, 504):
                        await asyncio.sleep(2 ** att * 3); continue
                    r.raise_for_status(); d = r.json()
                    text = "".join(c.get("text", "") for o in d.get("output", []) if o.get("type") == "message" for c in o.get("content", []))
                    return handle(it, text, d.get("usage", {}))
                except (httpx.HTTPError, ValueError) as e:
                    err = str(e)[:200]; await asyncio.sleep(2 ** att * 3)
            return handle(it, None, None, err)
    async with httpx.AsyncClient(timeout=300, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}) as client:
        return await asyncio.gather(*[one(client, it) for it in items])


def classify():
    todo = sorted({r["prompt"] for r in ROWS} - set(tracks())); print(len(todo), "prompts to classify", flush=True)
    body = lambda q: {"model": "gpt-6.1-sol", "reasoning": {"effort": "low"},
                      "input": [{"role": "developer", "content": PT.GUIDE}, {"role": "user", "content": f"# User request\n{q}"}],
                      "text": {"format": {"type": "json_schema", "name": "track", "strict": True, "schema": PT.SCHEMA}}}
    res = asyncio.run(post_all(todo, body, lambda q, t, u, e=None: {"prompt": q, **json.loads(t), "usage": u} if t else {"prompt": q, "error": e}))
    with open(TRACK_F, "a") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{sum('error' not in r for r in res)} classified, {sum('error' in r for r in res)} errors, ≈ ${cost([r['usage'] for r in res if r.get('usage')]):.2f};",
          Counter(tracks().values()))


def score():
    doc = open(f"{HERE}/prism_prompts_en.md").read()
    block = lambda h: re.search(rf"## {re.escape(h)}\n\n```\n(.*?)\n```", doc, re.S).group(1).replace("{{", "{").replace("}}", "}")
    AL = {t: block(f"alignment · {t}") for t in PT.TRACKS}; AE = block("aesthetic (all categories)"); T = tracks()
    done = {r["key"] for r in map(json.loads, open(CALLS_F)) if r.get("score") is not None} if os.path.exists(CALLS_F) else set()
    jobs = [{"key": f"{r['id']}|{dim}|{rd}", "row": r, "dim": dim} for r in ROWS for dim in ("alignment", "aesthetic") for rd in (1, 2, 3)
            if f"{r['id']}|{dim}|{rd}" not in done]
    if any(j["row"]["prompt"] not in T for j in jobs):
        raise SystemExit("run classify first: some prompts have no PRISM track")
    print(len(jobs), "PRISM calls", flush=True)
    score_of = lambda t: (lambda m: float(m.group(1)) if m else None)(re.search(r'"?score"?\s*[:：]\s*"?(\d+(?:\.\d+)?)', t or ""))
    f = open(CALLS_F, "a"); usage = []

    def body(j):
        r = j["row"]; tmpl = AL[T[r["prompt"]]] if j["dim"] == "alignment" else AE
        img = base64.b64encode(open(r["image"], "rb").read()).decode()
        return {"model": "gpt-6.1-sol", "reasoning": {"effort": "low"},
                "input": [{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": [{"type": "input_text", "text": tmpl.replace("{text_prompt}", r["prompt"])},
                                                       {"type": "input_image", "image_url": f"data:image/png;base64,{img}", "detail": "original"}]}]}

    def handle(j, text, u, err=None):
        rec = {"key": j["key"], "score": score_of(text), "text": text, "usage": u} if text is not None else {"key": j["key"], "error": err}
        if u:
            usage.append(u)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
    asyncio.run(post_all(jobs, body, handle)); f.close()
    print(f"{len(usage)} calls ok; ≈ ${cost(usage):.2f}")


def report():
    T, S = tracks(), defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(CALLS_F)):
        if r.get("score") is not None:
            i, dim, _ = r["key"].split("|"); S[i][dim].append(r["score"])
    by = defaultdict(dict); meta = {r["id"]: r for r in ROWS}
    for i, d in S.items():
        if i in meta and len(d["alignment"]) >= 3 and len(d["aesthetic"]) >= 3:
            al, ae = np.mean(d["alignment"][:3]), np.mean(d["aesthetic"][:3])
            by[meta[i]["group"]][meta[i]["case"]] = {"alignment": al, "aesthetic": ae, "combined": (al + ae) / 2, "track": T.get(meta[i]["prompt"])}
    groups = sorted(by); common = set.intersection(*[set(by[g]) for g in groups]) if groups else set()
    out = {"n_scored": {g: len(by[g]) for g in groups}, "n_common": len(common), "overall": {}, "by_track": {}, "paired": {}}
    for g in groups:
        out["overall"][g] = {m: round(float(np.mean([by[g][c][m] for c in common])), 4) for m in ("alignment", "aesthetic", "combined")}
        t = defaultdict(list)
        for c in common:
            t[by[g][c]["track"]].append(by[g][c]["combined"])
        out["by_track"][g] = {k: {"n": len(v), "combined": round(float(np.mean(v)), 4)} for k, v in sorted(t.items())}
    for i, x in enumerate(groups):
        for y in groups[i + 1:]:
            d = [by[y][c]["combined"] - by[x][c]["combined"] for c in common]
            out["paired"][f"{y} - {x}"] = {"mean_diff": round(float(np.mean(d)), 4), "se": round(float(np.std(d, ddof=1) / np.sqrt(len(d))), 4),
                                           "win": sum(v > .25 for v in d), "tie": sum(abs(v) <= .25 for v in d), "loss": sum(v < -.25 for v in d)}
    json.dump(out, open(f"{a.out}/prism_report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("n_scored", "n_common", "overall", "paired")}, ensure_ascii=False, indent=1))


{"classify": classify, "score": score, "report": report}[a.cmd]()
