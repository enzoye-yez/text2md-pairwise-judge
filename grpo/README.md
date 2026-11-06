# GRPO for the 4B text2md model with the terra v1b pairwise judge as reward

`train_grpo_md.py` trains the 4B text2md model (Qwen3.5, Gemini-10m SFT checkpoint 70450) with TRL `GRPOTrainer`:
vLLM colocated generation, DAPO loss, group-scaled advantages, vLLM importance-sampling correction.
The reward comes from [`pairwise_judge.py`](../pairwise_judge.py) (terra v1b) plus rule penalties, one scalar per rollout:

| Part | Rule |
|---|---|
| Format gate | Not the four sections in order, or truncated / no EOS → −1, not judged |
| Preference (Krea-3 style pivot) | One valid rollout of the group is the pivot; every other valid rollout is compared with it once by terra v1b (A/B order by hash). Reward = P(this better) − P(pivot better) ∈ [−1, 1]; the pivot gets 0. G−1 judge calls per group |
| Length | ratio = tokens / `ref_ntok` (mean SFT sampled length for the prompt): penalty below 0.85× and above 1.2× |
| Repetition | 8-gram repeat share above 0.2 is penalised |
| Requested text | Each quoted string of the request missing from the md −0.3 (cap −0.6); hedges like `"X" or "Y"` −0.15 each |
| Noise gate | Groups whose reward std < 0.1 get zero advantage (no update from judge noise) |

All knobs are command-line flags (see the top of the script).

## The run `grpo_g10m4b_pivot8_r1`

[`run_pivot8_r1.sh`](run_pivot8_r1.sh): 4 GPUs (ZeRO-2, [`zero2.yaml`](zero2.yaml)), 32 prompts × 8 rollouts per step,
lr 3e-6 constant, β (KL) 0.02, ε 0.2 / 0.28, temperature 1.0, checkpoint every 25 steps. 34,888 English prompts not in the
evaluation sets. Stopped at step 625 of 2,000. About 7 judge calls per prompt, ≈ $0.0026 per call.

Training curves (means per 50 steps): KL to the reference rose steadily from 0.017 to 0.124; mean completion length stayed
580–630 tokens; the share of groups with zero reward spread rose from 26% to 41%.

[`export_handoff.py`](export_handoff.py) turns a TRL checkpoint back into the layout of the original handoff checkpoint
(config, tokenizer, untrained MTP tensors copied from the base) so the unchanged text2md inference runtime loads it.

## Checkpoint evaluation

1. **md only, no rendering** — [`../eval/ckpt_vs_sft.py`](../eval/ckpt_vs_sft.py): greedy mds of the checkpoint vs the SFT
   model on PRISM 700, terra v1b in both A/B orders, margins averaged. Results so far (700 prompts each):

   | ckpt | win@0.6 | loss@0.6 | winrate (ties ½) | win / loss (sign) | median words |
   |---|---|---|---|---|---|
   | step25 | 5.0% | 5.1% | 0.499 | 25.7% / 43.9% | 334 |
   | step100 | 6.7% | 3.3% | 0.517 | 37.1% / 29.6% | 359 |
   | step175 | 8.6% | 3.1% | 0.527 | 38.6% / 30.0% | 358 |
   | step225 | 7.4% | 3.3% | 0.521 | 37.4% / 31.6% | 347 |
   | step325 | 7.3% | 3.3% | 0.520 | 39.4% / 27.6% | 373 |

   With 700 prompts the winrate is uncertain by about ±1–2 points, so the checkpoints after step 100 are not separable by this alone.

2. **Rendered images** — step175, step325 and step625 are rendered with the standard text2md evaluation pipeline
   (MD2JSON Text2JSON-27B step378 → J2I Ckpt17, same seeds and sizes as the SFT baseline), then scored with
   [`../eval/prism_score.py`](../eval/prism_score.py): PRISM with per-request tracks, 3 rounds averaged.
