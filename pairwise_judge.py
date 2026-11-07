"""Pairwise text2md judge, prompt v1b, on gpt-6.1-terra (OpenAI Responses API, reasoning effort low).

Two Markdown design briefs (A, B) for the same request; the model answers strict JSON
{reason, p_A_better, p_B_better, p_similar}. One call per pair: which md is shown as A is fixed by md5(pid) % 2, so
position bias does not line up with any side. margin = P(y better) - P(x better) in [-1, 1]; |margin| is the confidence.
Prompt versions (PROMPTS):
  v1b   PREAMBLE + GUARD + JSON_GUIDE. Used for the r45k DPO data (epochs 1-3) and as the GRPO reward (grpo/).
  v1b.1 v1b with one sentence added to the photo/illustration item of GUARD (and the previous sentence repeated):
        "Two valid artistic interpretations of the same request are similar." Used for everything from 2026-11-06 on:
        the 27B threshold study and the one-pair-per-prompt DPO picking in dpo_pick/ (10k pairs, 2026-11-07).
SYSTEM is v1b (the default everywhere); pass version="v1b.1" to reproduce dpo_pick.
"""
import asyncio, hashlib, json, os, random, time
import httpx

MODEL = "gpt-6.1-terra"
PRICE = (1.0, 0.10, 5.0)  # USD per 1M tokens: input, cached input, output

PREAMBLE = """You compare two Markdown design briefs, A and B, written for the same user request. Each brief is the only input to an image generation pipeline: a layout model turns it into objects and text, then an image model renders it. The pipeline never sees the request. Decide which brief will produce the better final image, as judged by a person with good visual taste who also checks the image against the request.

What matters, roughly in this order:
1. Requirements kept: every explicit requirement of the request (subjects, counts, attributes, required text, exclusions, format) is present and unchanged.
2. Text on the image: the words to render are exactly right (wording, spelling, numbers, one language, no stray characters from another writing system), and the amount of text suits the format.
3. Coherence: no contradictions or physically impossible combinations; the main subject is described concretely and plausibly.
4. Design quality: a clear layout and focal point, and a fitting style, palette and composition; added details support the request.

Expanding the request is the brief's job: added composition, lighting, props and styling are good when they fit. Length, richness and amount of detail are never good or bad by themselves; do not prefer a brief for being shorter or longer.

Answer similar unless one brief will clearly give the better image. Small differences in wording, ordering or amount of detail are similar. Only pick a brief when you are confident that a careful viewer would agree."""

JSON_GUIDE = """# How to answer
First write reason in Simplified Chinese: at most 60 characters naming the difference that decides the answer (or why they are similar). Then give three probabilities that sum to 1: p_A_better, p_B_better, p_similar, meaning how likely a careful viewer comparing the two final images would say A is clearly better, B is clearly better, or neither. Use p_similar for everything you are not confident about."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reason", "p_A_better", "p_B_better", "p_similar"],
          "properties": {"reason": {"type": "string"}, "p_A_better": {"type": "number"}, "p_B_better": {"type": "number"}, "p_similar": {"type": "number"}}}

GUARD = """

# Avoid these wrong preferences
- Length: do not prefer the shorter brief. A longer brief is not more likely to fail just because it has more elements, more text or more detail; judge whether those additions are correct and fit the request. Before answering, ask yourself: if the two briefs were the same length, would I still pick the same one? If not, answer similar.
- Plainness: do not reward a brief for being plain, generic or safe. A richer composition with props, context and styling is better when it fits the request and keeps the requested subject in charge.
- Photos, paintings and illustrations with little or no text: most of the image quality comes from how the subject and scene are depicted. Prefer the brief whose main subject, pose, anatomy, materials, lighting and spatial arrangement are concrete, physically plausible and coherent. Differences in wording, ordering or optional props are similar.
- Judge for a human viewer: do not favour a brief because it reads like a generic image-generation recipe (stacks of quality words such as cinematic, 8k, ultra-detailed, masterpiece). Concrete visual decisions matter; buzzwords do not."""

SYSTEM = PREAMBLE + GUARD + "\n\n" + JSON_GUIDE
_PHOTO_END = "Differences in wording, ordering or optional props are similar."
GUARD_V1B1 = GUARD.replace(_PHOTO_END, _PHOTO_END + " Two valid artistic interpretations of the same request are similar. " + _PHOTO_END, 1)
PROMPTS = {"v1b": SYSTEM, "v1b.1": PREAMBLE + GUARD_V1B1 + "\n\n" + JSON_GUIDE}
TRANSIENT = {408, 409, 429, 500, 502, 503, 504}


def user_text(prompt, ar, a, b):
    ar = f"\nTarget aspect ratio: {ar}" if ar else ""
    return f"# User request\n{prompt}{ar}\n\n# Brief A\n{a}\n\n# Brief B\n{b}\n"


def norm(p):
    p = {k: max(0.0, float(v)) for k, v in p.items()}; s = sum(p.values()) or 1.0
    return {k: v / s for k, v in p.items()}


def order_of(pid):  # 0: x is shown as A, 1: y is shown as A
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2


def margin(probs, order):  # + means y is better
    m = probs["B_better"] - probs["A_better"]
    return round(m if order == 0 else -m, 6)


def request_body(prompt, ar, md_a, md_b, version="v1b", effort="low"):
    """Responses API body for one comparison (md_a shown as Brief A); also used for Batch API request lines."""
    return {"model": MODEL, "reasoning": {"effort": effort},
            "input": [{"role": "developer", "content": PROMPTS[version]}, {"role": "user", "content": user_text(prompt, ar, md_a, md_b)}],
            "text": {"format": {"type": "json_schema", "name": "pair", "strict": True, "schema": SCHEMA}}}


def parse_body(body):
    """Responses API body -> normalised {A_better, B_better, similar} and the reason."""
    t = json.loads("".join(c.get("text", "") for o in body["output"] if o.get("type") == "message" for c in o["content"]))
    return norm({"A_better": t["p_A_better"], "B_better": t["p_B_better"], "similar": t["p_similar"]}), t["reason"]


async def call(client, pair, headers, effort="low", version="v1b"):
    order = order_of(pair["pid"])
    a, b = (pair["md_x"], pair["md_y"]) if order == 0 else (pair["md_y"], pair["md_x"])
    payload = request_body(pair["prompt"], pair.get("aspect_ratio"), a, b, version, effort)
    r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
    if r.status_code != 200:
        return r.status_code, None
    body = r.json(); probs, reason = parse_body(body)
    return 200, {"pid": pair["pid"], "order": order, "probs": probs, "margin": margin(probs, order), "reason": reason,
                 "usage": body.get("usage"), "model": body.get("model"), "prompt_version": version}


async def judge(client, sem, pair, headers, attempts=6, version="v1b"):
    err = None
    for att in range(attempts):
        async with sem:
            try:
                code, res = await call(client, pair, headers, version=version)
                if code == 200:
                    return res
                err = f"http_{code}"
                if code not in TRANSIENT:
                    break
            except (httpx.HTTPError, ValueError, KeyError, TypeError, asyncio.TimeoutError) as e:
                err = type(e).__name__
        await asyncio.sleep(min(60, 2 ** att + random.random()))
    return {"pid": pair["pid"], "error": err}


async def run(pairs, out_path, concurrency=24, version="v1b"):
    """Judge every pair not already in out_path (resumable); appends one JSON row per pair."""
    done = {d["pid"] for d in map(json.loads, open(out_path)) if "error" not in d} if os.path.exists(out_path) else set()
    todo = [p for p in pairs if p["pid"] not in done]
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    sem = asyncio.Semaphore(concurrency); t0 = time.time(); n = errs = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
        with open(out_path, "a") as f:
            for fut in asyncio.as_completed([judge(client, sem, p, headers, version=version) for p in todo]):
                r = await fut; f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); n += 1; errs += "error" in r
                if n % 200 == 0 or n == len(todo):
                    print(f"{n}/{len(todo)} done, {errs} errors, {time.time() - t0:.0f}s", flush=True)


def cost(rows):
    u = [d.get("usage") or {} for d in rows if "error" not in d]
    tin = sum(x.get("input_tokens", 0) for x in u); tc = sum((x.get("input_tokens_details") or {}).get("cached_tokens", 0) for x in u)
    tout = sum(x.get("output_tokens", 0) for x in u)
    return {"calls": len(u), "input": tin, "cached": tc, "output": tout, "usd": ((tin - tc) * PRICE[0] + tc * PRICE[1] + tout * PRICE[2]) / 1e6}
