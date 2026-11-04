# text2md pairwise judge (terra v1b)

Given one user request and two text2md Markdown briefs for it, predict which brief will give the better final image
(text2md md → Text2JSON → J2I), without rendering. Used to pick DPO pairs for the 4B text2md model.

- Model: `gpt-6.1-terra`, Responses API, reasoning effort `low`, strict JSON output `{reason, p_A_better, p_B_better, p_similar}`.
- Prompt: **v1b** = the base comparison prompt plus a guard paragraph against known wrong preferences (shorter, plainer,
  photo handling, recipe buzzwords). Full text in [`pairwise_judge.py`](pairwise_judge.py) (`PREAMBLE`, `GUARD`, `JSON_GUIDE`).
- **One call per pair.** Which md is shown as A is fixed by `md5(pid) % 2`, so position bias is not tied to either side.
- `margin = P(y better) − P(x better)` in [−1, 1]. Sign = the pick, `|margin|` = confidence. Keep a pair when `|margin| ≥ t`.

## Usage

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...            # never commit it
python judge_pairs.py examples/pairs.jsonl out.jsonl --threshold 0.8
python evaluate.py out.jsonl examples/labels.jsonl
```

Input (`pairs.jsonl`, one per line): `{"pid", "prompt", "aspect_ratio"?, "md_x", "md_y"}`.
Output rows: `{"pid", "order", "probs", "margin", "reason", "usage", "model"}`; `margin > 0` means `md_y` is better.
Runs are resumable (pairs already in the output file are skipped). Cost ≈ $0.0026 per pair for ~2k input tokens
(price constants in `pairwise_judge.py`).

`evaluate.py` takes labels `{"pid", "label": "x"|"y"|"similar"}` and reports, per threshold:
**coverage** (share of labelled pairs the judge calls), **precision** (share of those calls that agree with the label),
**false_call** (share of label-"similar" pairs it still calls), and **acc_all** (agreement over all labelled pairs).

## How well it works (labels = PRISM on rendered images; |PRISM gap| ≤ 0.25 counts as similar)

| Set | ≥ 0.6 coverage / precision | ≥ 0.8 coverage / precision |
|---|---|---|
| 27B rollouts, PRISM best vs worst of 8 (500 pairs, mean of 5 runs) | 39.8% / 79.4% | 22.1% / 88.8% |
| 27B rollout vs Gemini md (500 pairs, mean of 5 runs) | 46.3% / 72.4% | 29.8% / 80.2% |
| 27B rollouts, **random** 2 of 8 (1,000 pairs) | 37.9% / 70.7% | 17.7% / 83.5% |

- Repeated runs are stable (5 runs of the same pairs agree to within a few points).
- On random rollout pairs 35% are PRISM-similar; counting those as half-right, the share of correct DPO pairs is about
  65% / 67% / 76% at thresholds 0.6 / 0.7 / 0.8, so **0.8** is the recommended threshold for random-pair DPO data.
- First check on 4B (Gemini-5M) rollouts, rendered and PRISM-scored: |margin| ≥ 0.7 → 84% precision (42/50, 95% CI 74–94%).

## Known limits

- **Different writers:** when the two mds come from different sources (e.g. Gemini vs Sol), the judge prefers the longer,
  more detailed style; on Gemini-vs-Sol it cannot reach 70% precision. Use it on pairs from the same model.
- Longer / stricter prompt variants (v2–v4) only rescale the probabilities; at equal coverage they rank no better than v1b.
- The ~600-token system prompt is below the 1,024-token prompt-cache minimum, so it is not cached.
