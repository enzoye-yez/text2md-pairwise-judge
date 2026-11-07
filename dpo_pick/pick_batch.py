"""One DPO pair per prompt, at scale: luna checks (online Decisions, luna_checks.py) rank the candidate pairs, terra
(pairwise_judge.py, prompt v1b.1) on the OpenAI Batch API decides them. See dpo_pick/README.md. luna only ranks the candidate pairs (prior); it does NOT veto: at |margin| >= 0.8 the
luna veto removed 30% of passing 27B pairs whose image agreement was as high as the rest (0.83-0.94). Only the hard rules
(regex self-correction, unfinished, format-invalid) keep an md from being chosen. VETO_LUNA = True restores the veto.

Each round (R = 1: 8 rollouts, R = 2: 16 rollouts):
  luna R     luna on every new md that is not a hard negative
  a R        terra batch, FIRST order of up to MAX_TRY untried candidate pairs per open prompt
  b R        terra batch, SECOND order of the pairs whose first |margin| >= GATE
  collect    download finished batches into terra.jsonl (resumable, run until nothing is pending)
  decide R   per prompt keep the pair with the largest mean |margin| >= KEEP whose winner is not a hard negative -> picks.jsonl;
             R = 1 also writes the round-2 rollout shards (open prompts) for roll.sh
usage: pick_batch.py WORK_DIR luna|a|b|decide R  |  WORK_DIR collect|status
WORK_DIR holds rollouts under out/r{R}/g*/rollouts.jsonl (written by roll.sh) and the judge state under judge/.
"""
import asyncio, glob, hashlib, json, os, sys, time
import httpx

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import luna_checks as checkers  # noqa: E402
import pairwise_judge as pj  # noqa: E402

PROMPT = "v1b.1"  # terra prompt version used for the 2026-11-07 10k run
# luna alarm thresholds (P(major)) matched on 27B dev to the alarm rates of separate calls; only used when VETO_LUNA = True
ALARM = {"text": 0.68, "ending": 0.271, "short": 0.4, "align": 0.85, "aesth": 0.86}; VETO_LUNA = False
KEEP, GATE, MAX_TRY = 0.8, 0.6, 3
API = "https://api.openai.com/v1"


def hard_negative(r):
    return (not r.get("format_valid", True)) or r.get("finish_reason", "stop") != "stop" or not r["markdown"].strip() \
        or bool(checkers.LOOP_RE.search(r["markdown"]))


def first_order(pid):
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2


async def online(reqs, concurrency=32):
    """Batch request lines -> terra rows via the normal Responses API (retries on transient errors)."""
    sem = asyncio.Semaphore(concurrency); headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    async def one(cl, r):
        pid, o = r["custom_id"].rsplit("#", 1)
        for att in range(6):
            async with sem:
                try:
                    resp = await cl.post(f"{API}/responses", json=r["body"])
                    if resp.status_code == 200:
                        bd = resp.json(); probs, reason = pj.parse_body(bd)
                        return {"pid": pid, "order": int(o), "probs": probs, "reason": reason, "usage": bd.get("usage"), "batch": "online"}
                    if resp.status_code not in pj.TRANSIENT:
                        return None
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    pass
            await asyncio.sleep(min(60, 2 ** att))
        return None
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10), headers=headers) as cl:
        res = await asyncio.gather(*[one(cl, r) for r in reqs])
    return [r for r in res if r]


class Work:
    def __init__(self, w):
        self.w = w; self.j = f"{w}/judge"; os.makedirs(self.j, exist_ok=True)
        self.luna_path, self.terra_path, self.picks_path = f"{self.j}/luna_m1.jsonl", f"{self.j}/terra.jsonl", f"{self.j}/picks.jsonl"

    def rollouts(self, upto):
        R = {}
        for r in range(1, upto + 1):
            for f in sorted(glob.glob(f"{self.w}/out/r{r}/g*/rollouts.jsonl")):
                for l in open(f):
                    d = json.loads(l); d["k"] = d["k"] + 8 * (r - 1)  # round 2 samples are k 8..15
                    R[f"{d['id']}|{d['k']}"] = d
        by = {}
        for k, d in R.items():
            by.setdefault(d["id"], []).append(k)
        return R, by

    def luna(self):
        return checkers.load_merged(self.luna_path) if os.path.exists(self.luna_path) else {}

    def terra(self):
        T = {}
        for d in map(json.loads, open(self.terra_path)) if os.path.exists(self.terra_path) else []:
            if "error" not in d:
                pr = d["probs"]; T.setdefault(d["pid"], {})[d["order"]] = (pr["B_better"] - pr["A_better"]) * (1 if d["order"] == 0 else -1)
        return T  # + = y better

    def picked(self):
        return {json.loads(l)["id"] for l in open(self.picks_path)} if os.path.exists(self.picks_path) else set()

    # -- steps -------------------------------------------------------------------------------------------------------
    def run_luna(self, rnd):
        R, _ = self.rollouts(rnd); have = self.luna(); done = self.picked()
        items = [{"key": k, "prompt": r["prompt"], "aspect_ratio": r.get("aspect_ratio"), "md": r["markdown"]}
                 for k, r in R.items() if k not in have and r["id"] not in done and not hard_negative(r)]
        print(len(items), "mds for luna", flush=True)
        asyncio.run(checkers.run_merged(items, self.luna_path, 96))

    def helpers(self, R, L):
        """prior / alarm per md, computed once (re-running the regex inside every comparison was far too slow at 192k mds).
        Only mds that are hard negatives or have luna scores get a prior."""
        HN = {k: hard_negative(r) for k, r in R.items()}
        PRI = {k: 99.0 if HN[k] else sum(L[k].values()) for k in R if HN[k] or k in L}
        AL = {k: HN[k] or (VETO_LUNA and any(L[k][c] >= t for c, t in ALARM.items())) for k in PRI}
        return PRI, AL

    def candidates(self, rnd):
        R, by = self.rollouts(rnd); L = self.luna(); T = self.terra(); done = self.picked()
        PRI, AL = self.helpers(R, L)
        out = {}
        for i, ks in by.items():
            if i in done or not all(k in PRI for k in ks):
                continue
            clean = {k for k in ks if not AL[k]}
            pairs = sorted(((PRI[o] - PRI[c], c, o) for c in clean for o in ks if o != c and (o not in clean or (PRI[o], o) > (PRI[c], c))), reverse=True)
            out[i] = [(f"{c}|{o.split('|')[-1]}", c, o) for _, c, o in pairs]
        return R, out, T

    def submit(self, reqs, tag):
        """reqs: [(pid, x_key, y_key, order)] -> batch files of <= 12,000 requests; ids appended to judge/batches.json."""
        R, _ = self.rollouts(2)
        reg_path = f"{self.j}/batches.json"; reg = json.load(open(reg_path)) if os.path.exists(reg_path) else []
        with httpx.Client(timeout=600, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}) as c:
            for s in range(0, len(reqs), 12000):
                fn = f"{self.j}/in_{tag}_{s // 12000}.jsonl"
                with open(fn, "w") as f:
                    for pid, x, y, o in reqs[s:s + 12000]:
                        A, B = (R[x], R[y]) if o == 0 else (R[y], R[x])
                        body = pj.request_body(A["prompt"], A.get("aspect_ratio"), A["markdown"], B["markdown"], PROMPT)
                        f.write(json.dumps({"custom_id": f"{pid}#{o}", "method": "POST", "url": "/v1/responses", "body": body}, ensure_ascii=False) + "\n")
                up = c.post(f"{API}/files", data={"purpose": "batch"}, files={"file": (os.path.basename(fn), open(fn, "rb"))}); up.raise_for_status()
                b = c.post(f"{API}/batches", json={"input_file_id": up.json()["id"], "endpoint": "/v1/responses", "completion_window": "24h"}); b.raise_for_status()
                reg.append({"tag": tag, "file": fn, "n": len(reqs[s:s + 12000]), "batch_id": b.json()["id"], "collected": False})
                json.dump(reg, open(reg_path, "w"), indent=1)
                print(f"{fn}: {reg[-1]['n']} requests, {os.path.getsize(fn) / 1e6:.0f} MB -> {reg[-1]['batch_id']}", flush=True)

    def stage_a(self, rnd):
        R, cand, T = self.candidates(rnd); reqs = []
        for i, pairs in cand.items():
            new = [(pid, c, o) for pid, c, o in pairs if pid not in T][:MAX_TRY]
            reqs += [(pid, c, o, first_order(pid)) for pid, c, o in new]
        print(f"{len(cand)} open prompts, {len(reqs)} first-order requests", flush=True)
        self.submit(reqs, f"r{rnd}a")

    def stage_b(self, rnd):
        R, cand, T = self.candidates(rnd); reqs = []
        for i, pairs in cand.items():
            for pid, c, o in pairs:
                t = T.get(pid, {})
                if len(t) == 1 and abs(next(iter(t.values()))) >= GATE:
                    reqs.append((pid, c, o, 1 - next(iter(t))))
        print(f"{len(reqs)} second-order requests", flush=True)
        self.submit(reqs, f"r{rnd}b")

    def collect(self):
        reg_path = f"{self.j}/batches.json"; reg = json.load(open(reg_path)); pending = 0
        with httpx.Client(timeout=600, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}) as c, open(self.terra_path, "a") as out:
            for x in reg:
                if x["collected"]:
                    continue
                b = c.get(f"{API}/batches/{x['batch_id']}").json()
                if b["status"] not in ("completed", "cancelled", "expired", "failed"):
                    rc = b.get("request_counts") or {}
                    if b["status"] == "in_progress" and rc.get("total") and rc.get("completed", 0) >= .99 * rc["total"]:
                        if "at99" not in x:  # first seen at 99%: start the 30-minute clock
                            x["at99"] = time.time(); json.dump(reg, open(reg_path, "w"), indent=1)
                        elif time.time() - x["at99"] > 1800:  # still not done 30 min after 99%: cancel, the rest goes online
                            c.post(f"{API}/batches/{x['batch_id']}/cancel")
                    pending += 1; print(x["batch_id"], x["tag"], b["status"], rc); continue
                n = errs = 0; got = set()
                for k in ("output_file_id", "error_file_id"):
                    if not b.get(k):
                        continue
                    for l in c.get(f"{API}/files/{b[k]}/content").text.splitlines():
                        d = json.loads(l); pid, o = d["custom_id"].rsplit("#", 1); r = d.get("response") or {}; bd = r.get("body") or {}
                        try:
                            assert r.get("status_code") == 200
                            probs, reason = pj.parse_body(bd)
                            row = {"pid": pid, "order": int(o), "probs": probs, "reason": reason, "usage": bd.get("usage"), "batch": x["batch_id"]}
                            n += 1; got.add(d["custom_id"])
                        except (AssertionError, KeyError, ValueError, TypeError):
                            row = {"pid": pid, "order": int(o), "error": r.get("status_code") or d.get("error")}; errs += 1
                        if "error" not in row:
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                todo = [r for r in map(json.loads, open(x["file"])) if r["custom_id"] not in got]
                if todo:  # cancelled tail or errors: the same requests on the normal API
                    rows = asyncio.run(online(todo)); out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows); out.flush()
                    print(f"{x['batch_id']}: {len(todo)} requests sent online, {len(rows)} ok", flush=True)
                x["collected"] = True; json.dump(reg, open(reg_path, "w"), indent=1)
                print(f"{x['batch_id']} {x['tag']} {b['status']}: {n} rows, {errs} errors", flush=True)
        print(f"pending batches: {pending}")
        return pending

    def decide(self, rnd):
        R, cand, T = self.candidates(rnd); L = self.luna(); PRI, AL = self.helpers(R, L)
        _, by = self.rollouts(rnd); n = 0
        with open(self.picks_path, "a") as f:
            for i, pairs in cand.items():
                best = None
                for pid, c, o in pairs:
                    t = T.get(pid, {})
                    if len(t) == 2:
                        m = sum(t.values()) / 2; win, lose = (o, c) if m > 0 else (c, o)
                        if abs(m) >= KEEP and not AL[win] and (best is None or abs(m) > best[0]):
                            best = (abs(m), win, lose)
                if best:
                    m, win, lose = best; n += 1
                    f.write(json.dumps({"id": i, "prompt": R[win]["prompt"], "aspect_ratio": R[win].get("aspect_ratio"),
                                        "chosen": R[win]["markdown"], "rejected": R[lose]["markdown"], "k_chosen": R[win]["k"], "k_rejected": R[lose]["k"],
                                        "confidence": round(m, 4), "prior_chosen": round(PRI[win], 4), "prior_rejected": round(PRI[lose], 4),
                                        "rejected_hard_negative": hard_negative(R[lose]), "words_chosen": len(R[win]["markdown"].split()),
                                        "words_rejected": len(R[lose]["markdown"].split()), "n_rollouts": len(by[i]),
                                        "round": rnd}, ensure_ascii=False) + "\n")
        done = self.picked()
        open_ids = [i for i in by if i not in done]
        print(f"round {rnd}: {n} new picks, {len(done)} picked in total, {len(open_ids)} of {len(by)} prompts open")
        if rnd == 1 and not glob.glob(f"{self.w}/out/r2/g*/rollouts.jsonl"):
            src = {json.loads(l)["id"]: l for f in glob.glob(f"{self.w}/shards/r1_g*.jsonl") for l in open(f)}
            for g in range(8):
                open(f"{self.w}/shards/r2_g{g}.jsonl", "w").writelines(src[i] for i in open_ids[g::8])
            print(f"wrote shards/r2_g*.jsonl for {len(open_ids)} prompts")


if __name__ == "__main__":
    W = Work(sys.argv[1]); cmd = sys.argv[2]; rnd = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    {"luna": lambda: W.run_luna(rnd), "a": lambda: W.stage_a(rnd), "b": lambda: W.stage_b(rnd), "collect": W.collect,
     "status": W.collect, "decide": lambda: W.decide(rnd)}[cmd]()
