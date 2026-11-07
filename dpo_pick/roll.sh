#!/bin/bash
# Roll 8 samples per prompt with the 4B text2md model, one docker container per GPU.
# usage: WORK=/path/to/work ROUND=1 SEED=2026110700 ./roll.sh
#   WORK/shards/r${ROUND}_g{i}.jsonl ({id, prompt}) -> WORK/out/r${ROUND}/g{i}/rollouts.jsonl
# MODEL_DIR: the checkpoint from Aran's handoff package (hf/ + model.json); CODE_DIR: that package's code/ (infer.py,
# scripts/) with sample.py from this folder copied in; IMAGE: the pinned vLLM 0.28.0 runtime from the same package.
# Sampling as in the r45k DPO data: n=8, temperature 0.7, top_p 0.95; seed = SEED + GPU index (use a new SEED per round).
set -eu
: "${WORK:?}" "${ROUND:?}" "${SEED:?}"
MODEL_DIR=${MODEL_DIR:-$HOME/data/models/aran_handoff/models/gemini-10m-4b-70450}
CODE_DIR=${CODE_DIR:-$HOME/text2md_dpo/r45k/code}
IMAGE=${IMAGE:-text2md-runtime:vllm0.28.0-20261028}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}; NAME=${NAME:-t2mdpo-pick}
for g in $GPUS; do
  [ -s "$WORK/shards/r${ROUND}_g$g.jsonl" ] || continue
  mkdir -p "$WORK/out/r$ROUND/g$g"
  docker run -d --name "$NAME-r$ROUND-g$g" --device nvidia.com/gpu=$g --ipc=host --network=none --entrypoint python3 \
    -v "$MODEL_DIR":/model:ro -v "$WORK/shards/r${ROUND}_g$g.jsonl":/input.jsonl:ro -v "$WORK/out/r$ROUND/g$g":/output -v "$CODE_DIR":/code:ro \
    -e PYTHONPATH=/code -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 "$IMAGE" \
    /code/sample.py --model-dir /model --input /input.jsonl --output /output --n 8 --temperature 0.7 --top-p 0.95 \
    --seed $((SEED + g)) --max-tokens 6144 --max-num-seqs 256 --chunk 256
done
