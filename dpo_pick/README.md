# dpo_pick: one DPO pair per prompt (luna ranks, terra decides)

Builds DPO pairs for the 4B text2md model from its own rollouts. For every prompt we roll several mds and look for ONE
pair whose preference is clear enough to show up in the rendered image. Taking one pair per prompt keeps the prompt
mix close to the pool; we keep rolling (8, then 16 rollouts) instead of lowering the bar.

## How a pair is picked

```
prompt ── roll 8 (4B SFT, T=0.7, top_p=0.95) ──┐
                                               ├─ hard rules ── hard negatives: can only be "rejected"
                                               ├─ luna (5 checks, 1 call per md) ── prior per md
                                               └─ up to 3 candidate pairs (cleanest md vs largest prior gap)
                                                     └─ terra v1b.1, order 1 ── |m| < 0.6 → drop
                                                           └─ terra, order 2 ── mean |m| ≥ 0.8 → keep (one per prompt)
no pair → roll 8 more (16 total) and repeat → still none: the prompt is dropped (logged)
```

1. **Hard rules (no model):** an md is a hard negative when it is not format-valid, did not finish (`finish_reason !=
   stop`, e.g. a loop hitting the token limit), or contains a self-correction note (`-> wait`, `is not used;`,
   `exact copy:` … see `LOOP_RE` in `luna_checks.py`). These come from the SFT data and turn into endless loops under
   greedy decoding. A hard negative is never chosen; it is tried first as the rejected side.
2. **luna prior (`luna_checks.py`):** one `gpt-6-luna` Decisions call per md with five questions (text copied wrongly,
   not ending properly, too thin, clear aesthetic mistake, inconsistent with the request), each none / minor / major.
   prior = sum of P(major). luna **only orders** the candidate pairs; it does not decide or veto.
3. **terra decides (`../pairwise_judge.py`, prompt `v1b.1`):** both A/B orders on the Batch API. First order for up to 3
   candidate pairs per prompt; if |margin| ≥ 0.6 the second order is asked; the pair is kept when the mean
   |margin| ≥ 0.8 and the winner is not a hard negative. The direction is terra's (the winner may be either md).
   Confidence of a kept pair = mean |margin|.

## Why these choices (Aran's 27B × 8 rollouts, labels = track-specific PRISM image scores)

Tuned on a DEV half of the 500 queries, checked on the TEST half; PRISM 700 / VQA were never used for tuning.

| terra mean \|margin\| | image agrees with terra | big wrong calls (image says the other md, gap ≥ 1) |
|---|---|---|
| ≥ 0.8 | **0.85–0.86** | 6–7% |
| 0.7–0.8 | 0.65 | 11% |
| 0.6–0.7 | 0.61 | 15% |
| < 0.6 | 0.51–0.59 | ~10% |

Only ≥ 0.8 is reliable, hence the threshold.

| luna's role | result | decision |
|---|---|---|
| Veto (drop the pair when terra's winner raises a luna alarm) | at \|m\| ≥ 0.8 it removed 30% of passing pairs whose image agreement was as high (0.83–0.94); no precision gain | **not used** |
| "Relaxed" bar when the loser raises an alarm (keep at \|m\| ≥ 0.3) | 0.88–0.93 on DEV, 0.55–0.61 on TEST | **not used** |
| Prior for choosing which pairs to send to terra | top-1 pair passes 19.0% vs 12.8% random; top-3 29.2% vs 24.9% | **used** |

Is terra a better prior than luna? Control run (terra scoring each md alone, same five checks + an overall 0–10):

| prior | top-1 pass | top-3 pass | image agreement of the picks |
|---|---|---|---|
| random | 12.7% | 25.0% | — |
| luna, 5 checks in one call | 19.0% | 29.2% | 0.850 |
| terra, 5 checks | 21.4% | 29.2% | 0.805 |
| terra, overall score | 22.6% | 30.2% | 0.851 |

terra helps a little at top-1 and not at top-3 (SE ≈ 2 points), costs ~6× more per md, and its pointwise score agrees
with its own pairwise winner 97.6% of the time (luna 79.6%), i.e. it mostly pre-guesses terra's answer. luna stays.

A dry run of the whole picker on the 27B data (8 rollouts per query, 3 tries): 29% of queries got a pair; the picks
agree with the image 0.836 (mean image gap +1.07).

## The 2026-11-07 run (4B SFT gemini-10m-4b-70450, prompts from the 45k r45k pool)

| | prompts | pairs | share |
|---|---|---|---|
| round 1 (8 rollouts) | 24,000 | 5,778 | 24% |
| round 2 (16 rollouts) | 18,222 open | 2,006 | 11% |
| total | 24,000 | **7,784** | 32% |

- A supplementary round-1 run on 10,400 unused prompts of the same pool was started to reach 10k pairs.
- Median confidence 0.885; chosen/rejected word ratio median 0.94 (chosen ≥ 15% shorter in 30% of pairs, ≥ 15% longer
  in 14%); 133 rejected mds are hard negatives.
- Prompt mix shift: picked prompts contain quoted text 14.6% (open prompts 9.2%) and non-Latin text 2.7% (0.5%): these
  prompts more often produce a broken rollout, so a clear pair is easier to find.
- Rollout speed: ~450–520 rollouts / min / GPU (RTX PRO 6000), 8 GPUs → 192k rollouts in ~55 min.
- API cost: luna ≈ $81 (~2.5k input tokens per md, 320k mds), terra Batch ≈ $215 (163k calls at half price). Batch
  turnaround 20–60 min per stage; tails are cancelled 30 min after 99% and finished on the normal API.

## Run it

```bash
export OPENAI_API_KEY=...                       # never commit it
WORK=/data/pick && mkdir -p $WORK/shards       # split the prompts: $WORK/shards/r1_g{0..7}.jsonl, rows {id, prompt}
WORK=$WORK SEED1=2026110700 SEED2=2026110800 nohup setsid ./pipeline.sh > $WORK/pipeline.log 2>&1 &
```

Steps can also be run by hand: `python pick_batch.py $WORK luna|a|b|decide ROUND` and `python pick_batch.py $WORK collect`.
Output: `$WORK/judge/picks.jsonl`, one row per prompt: `{id, prompt, aspect_ratio, chosen, rejected, k_chosen,
k_rejected, confidence, prior_chosen, prior_rejected, rejected_hard_negative, words_chosen, words_rejected, n_rollouts,
round}`. Raw calls: `judge/luna_m1.jsonl`, `judge/terra.jsonl`, batch registry `judge/batches.json`.

Rollouts use Aran's text2md handoff package (checkpoint + pinned vLLM 0.28.0 image + `code/`); copy `sample.py` from
this folder into that `code/` directory and point `roll.sh` at it (`MODEL_DIR`, `CODE_DIR`, `IMAGE`).

| File | What |
|---|---|
| `pick_batch.py` | candidate pairs, Batch submission / collection (with online fallback), per-prompt decision |
| `luna_checks.py` | luna prompt (five checks, one call), Decisions client, `LOOP_RE` hard rule |
| `roll.sh`, `sample.py` | 8 rollouts per prompt per GPU in the pinned runtime |
| `pipeline.sh` | both rounds end to end |
