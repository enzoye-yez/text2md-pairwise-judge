"""luna checks for DPO pair picking: five pointwise checks in ONE gpt-6-luna Decisions call (version m1), plus the regex
for self-correction notes used as a hard rule.

Each md is judged on its own (request + one brief). Questions: text (required text copied wrongly), ending (not finished,
loops, self-correction notes, garbage), short (requested content dropped / too thin), aesth (clear aesthetic mistakes),
align (inconsistent with the request); choices none / minor / major with probabilities. The prior used by pick_batch.py is
the sum of P(major) over the five checks (lower = cleaner). It only orders the candidate pairs; terra decides the pair.
Rows: {key, version, probs: {check: {none, minor, major}}, usage}; resumable on key.
Needs OPENAI_API_KEY. The Decisions endpoint is not available on the Batch API, so these calls are online (~2.5k input
tokens per md, about $0.00025 per md at $0.10/M).
"""
import asyncio, json, os, random, re, time
import httpx

BASE = "https://api.openai.com/v1"
MODEL = os.environ.get("LUNA_MODEL", "gpt-6-luna")
TRANSIENT = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}  # 52x: Cloudflare edge errors, transient

PREAMBLE = """You check ONE Markdown design brief written for a user's image request. The brief is the only input to an image pipeline that draws exactly what the brief says; the pipeline never sees the request. You look for one kind of clear mistake only, described in the question. Ignore every other quality of the brief. Most briefs are fine: answer none unless the mistake is really there and you can point to it."""

SEV = [{"value": "none", "description": "The mistake is not there (or this check does not apply to this request)."},
       {"value": "minor", "description": "The mistake is there but small: a viewer of the final image would hardly notice it."},
       {"value": "major", "description": "The mistake is clearly there and a viewer of the final image would notice it."}]

CHECKS = {
    "text": """Required text copied wrongly. First find the text the request asks to appear in the image: quoted strings, titles, headings, labels, names, slogans, numbers, prices, dates. If the request asks for no specific text, answer none. Otherwise compare each required string, character by character, with the text the brief tells the pipeline to print. Count as a mistake: a misspelled, re-worded, translated, transliterated or truncated string; a changed letter case where the request is explicit; changed digits, units or symbols (such as ™, &, %, currency signs); punctuation added or dropped (a period or comma from the request's own sentence printed after a short title, label or name: the request 'a sign that says "OPEN 24 HOURS".' asks for OPEN 24 HOURS; but '!' or '?' inside a slogan, 'CO., LTD.' and the period of a quoted full sentence belong to the text); a required string missing, printed twice, or split across places. A large unrequested headline that competes with the required text is minor. Major = a viewer would read a wrong or missing required string.""",
    "ending": """The brief does not end properly. Count as a mistake: the brief stops in the middle of a sentence, list or section, or a section it announces is missing; the same sentence, list item or phrase is repeated over and over; the brief contains self-correction or working notes instead of a clean description (for example '-> wait', 'no, exact copy:', 'X is not used; the exact lettering reads Y', 'corrected to', 'let me fix'), or other leftover reasoning, meta comments or garbage characters. A brief that is short but complete and clean is NOT a mistake here. Major = the pipeline would draw garbled, duplicated or missing parts because of it.""",
    "short": """Too thin for the request: information the image needs is lost. Look at what the request explicitly asks for (items, people, objects, panels, sections, rows, data points, steps, text blocks, details of the subject, setting, style). Count as a mistake: requested content dropped; several requested items merged or summarised into a vague phrase ('various icons', 'some text', 'dates 1 to 31') instead of being specified; a long or detailed request answered with a brief so short that the layout, arrangement or look of the main subject is left to guesswork. Extra detail the request did not ask for is NOT needed; do not count a concise brief that still covers everything requested. Major = a viewer comparing the image with the request would notice missing or vague content.""",
    "aesth": """A clear aesthetic mistake the brief commits the image to. Count as a mistake: so many elements or text blocks that the image will be cluttered or cramped; important text that will be too small, low contrast or placed over a busy background; a palette or style mix that clashes with itself or the request's mood; no clear focal point or hierarchy; the main subject awkwardly cropped, tiny, or off balance for the aspect ratio; a design that will look cheap, generic or amateurish for its purpose (a poster, cover, menu, slide, logo). Do NOT count a style simply because you would choose another one; only count choices that would make most viewers find the image worse. Major = most viewers would find the image clearly unattractive or hard to read because of it.""",
    "align": """Inconsistent with the request. Count as a mistake: the brief changes what the request asks for: a different main subject, entity, character, brand or setting; a different medium or style (a photo when an illustration is asked, or the reverse); a wrong count, size, position, relation, color or action the request states; a different format or layout (a poster asked, a scene given; a grid of 4 asked, 6 given); a required element missing; an unrequested element that takes over the image; or two parts of the brief that contradict each other. Reasonable additions that serve the request are NOT mistakes, and do not judge text spelling here. Major = a viewer comparing the image with the request would say it does not show what was asked.""",
}


# Self-correction notes the 4B model leaves in its output (from the SFT data); an md that matches is a hard negative.
LOOP_RE = re.compile(r"->\s*wait|\bwait,? no\b|exact copy:|is not used;|is omitted;|corrected to exact|let me (fix|correct)", re.I)

MERGED_PREAMBLE = PREAMBLE.replace("You look for one kind of clear mistake only, described in the question. Ignore every other quality of the brief.",
                                   "Answer each question on its own: each looks for one kind of clear mistake only and ignores every other quality of the brief.")


async def judge_merged(client, sem, it, attempts=8):
    payload = {"model": MODEL, "input": f"{MERGED_PREAMBLE}\n\n# User request\n{it['prompt']}" + (f"\nTarget aspect ratio: {it['aspect_ratio']}" if it.get("aspect_ratio") else "") + f"\n\n# Brief\n{it['md']}\n",
               "questions": [{"name": c, "type": "choice", "instructions": CHECKS[c], "choices": SEV} for c in CHECKS]}
    err = None
    for att in range(attempts):
        async with sem:
            try:
                r = await client.post(f"{BASE}/decisions", json=payload)
                if r.status_code == 200:
                    b = r.json(); out = {}
                    for c, a in zip(CHECKS, b["answers"]):
                        if a.get("name") != c or a.get("type") == "refusal":
                            raise ValueError("bad answer")
                        p = {e["value"]: float(e["probability"]) for e in a.get("probabilities") or []}
                        if set(p) != {"none", "minor", "major"}:
                            raise ValueError("bad choices")
                        out[c] = p
                    return {"key": it["key"], "version": "m1", "probs": out, "usage": b.get("usage")}
                err = f"http_{r.status_code}"
                if r.status_code not in TRANSIENT:
                    return {"key": it["key"], "error": f"{err}:{r.text[:200]}"}
            except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as e:
                err = type(e).__name__
        await asyncio.sleep(min(60, 2 ** att + random.random()))
    return {"key": it["key"], "error": err}


async def run_merged(items, out_path, concurrency=64):
    """items: {key, prompt, aspect_ratio?, md}; resumable on key."""
    done = {d["key"] for d in map(json.loads, open(out_path)) if "error" not in d} if os.path.exists(out_path) else set()
    todo = [it for it in items if it["key"] not in done]
    sem = asyncio.Semaphore(concurrency); t0 = time.time(); n = errs = 0
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                                 timeout=httpx.Timeout(120, connect=10), limits=httpx.Limits(max_connections=concurrency)) as client:
        with open(out_path, "a") as f:
            for fut in asyncio.as_completed([judge_merged(client, sem, it) for it in todo]):
                r = await fut; f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); n += 1; errs += "error" in r
                if n % 5000 == 0 or n == len(todo):
                    print(f"luna merged: {n}/{len(todo)} done, {errs} errors, {time.time() - t0:.0f}s", flush=True)


def load_merged(path):
    """-> {key: {check: P(major)}}"""
    return {d["key"]: {c: p["major"] for c, p in d["probs"].items()} for d in map(json.loads, open(path)) if "error" not in d}
