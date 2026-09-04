#!/bin/bash
# Benchmark a checkpoint that already exists.
#
#   sbatch scripts/jobs/bench.sh vqcfull_n2l5
#   sbatch scripts/jobs/bench.sh vqcfull_n2l5 checkpoints/.../params_ep050.pt
#
# Without a second argument it takes the run's final checkpoint. Independent of
# each other, so submit one per configuration and they run in parallel rather
# than queueing behind a single long job.
#SBATCH --job-name=bench
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

CFG=${1:?usage: sbatch bench.sh CONFIG [CHECKPOINT]}
ROOT=/cephfs/mbandhu/qulip

cd "$ROOT" || exit 1
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
mkdir -p results

RUN=$(grep -m1 'name:' "configs/$CFG.yaml" | tr -d ' "' | cut -d: -f2)
CKPT=${2:-$(ls -t checkpoints/"$RUN"/*/*/last.pt 2>/dev/null | head -1)}
[ -z "$CKPT" ] && { echo "no checkpoint for $RUN"; exit 1; }

TAG=$(basename "$CKPT" .pt)
echo "config $CFG | checkpoint $CKPT"
python -u -m scripts.benchmark -cfg "configs/$CFG.yaml" -cp "$CKPT" \
    2>&1 | tee "results/bench_${CFG}_${TAG}.txt"
