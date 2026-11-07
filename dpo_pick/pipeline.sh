#!/bin/bash
# Full run: round 1 (8 rollouts) and round 2 (8 more for the prompts without a pair), each:
# wait for rollouts -> luna -> terra stage a (Batch) -> collect -> stage b (Batch) -> collect -> decide.
# usage: WORK=/path/to/work SEED1=... SEED2=... nohup setsid ./pipeline.sh > $WORK/pipeline.log 2>&1 &
#   WORK/shards/r1_g{i}.jsonl must exist (prompts split over the GPUs); ROUNDS="1" stops after round 1.
# OPENAI_API_KEY must be in the environment. Collect polls every 5 min; a batch still running 30 min after it reached
# 99% is cancelled and its remaining requests are sent to the normal API.
set -u
: "${WORK:?}"; export WORK
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python3}; J="$PY $HERE/pick_batch.py $WORK"; NAME=${NAME:-t2mdpo-pick}
wait_roll() { while [ -n "$(docker ps -q -f name=$NAME-r$1-)" ]; do sleep 30; done; }
wait_batches() { until $J collect | tee -a "$WORK/collect.log" | grep -q "pending batches: 0"; do sleep 300; done; }
for R in ${ROUNDS:-1 2}; do
  if [ -z "$(docker ps -aq -f name=$NAME-r$R-)" ] && [ ! -d "$WORK/out/r$R" ]; then
    S=SEED$R; ROUND=$R SEED=${!S:?} "$HERE/roll.sh"
  fi
  wait_roll $R
  echo "== round $R: luna $(date)"; $J luna $R || exit 1
  echo "== round $R: terra stage a $(date)"; $J a $R || exit 1; wait_batches
  echo "== round $R: terra stage b $(date)"; $J b $R || exit 1; wait_batches
  echo "== round $R: decide $(date)"; $J decide $R || exit 1
done
echo "PIPELINE_DONE $(date)"
