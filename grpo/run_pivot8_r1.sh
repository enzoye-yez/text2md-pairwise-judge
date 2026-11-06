#!/bin/bash
# The run grpo_g10m4b_pivot8_r1 (2026-11-05): 4B Gemini-10m SFT ckpt70450, 4 x 96 GB GPUs (DeepSpeed ZeRO-2, vLLM colocated),
# 32 prompts x 8 rollouts per step, lr 3e-6 constant (5 warmup), KL beta 0.02, checkpoint every 25 steps.
# Stopped by hand at step 625 of the planned 2000. Other reward / length / repetition settings are the script defaults.
#   MODEL  HF dir of the SFT model         DATA  jsonl {"id", "messages": [system, user], "ref_ntok"}   OUT  run dir
set -e
MODEL=${MODEL:?HF model dir}; DATA=${DATA:?prompt jsonl}; OUT=${OUT:-runs/grpo_g10m4b_pivot8_r1}
: "${OPENAI_API_KEY:?needed for the terra judge}"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} accelerate launch --config_file zero2.yaml --num_processes 4 train_grpo_md.py \
  --model "$MODEL" --data "$DATA" --out "$OUT" --steps 2000 --prompts-per-step 32 --group 8 --lr 3e-6 --beta 0.02 --save-steps 25 \
  > "$OUT/train.log" 2>&1
