#!/bin/bash
# Train one configuration, then benchmark the final checkpoint.
#
#   sbatch scripts/jobs/train.sh vqcfull_n2l5
#   sbatch scripts/jobs/train.sh vqcfull_n4l4 900
#
# Memory defaults by depth, since the tensor-ring bond dimension is 2^L and the
# test pass holds all 24,909 captions at once. Override with the second
# argument if a run turns out to need more.
#SBATCH --job-name=train
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --time=10-00:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

CFG=${1:?usage: sbatch train.sh CONFIG [MEMORY_GB]}
ROOT=/cephfs/mbandhu/qulip

cd "$ROOT" || exit 1
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

RUN=$(grep -m1 'name:' "configs/$CFG.yaml" | tr -d ' "' | cut -d: -f2)

echo "config $CFG | run $RUN | node $(hostname) | started $(date)"
python -u -m scripts.train -cfg "configs/$CFG.yaml" || exit 1

# The final checkpoint rather than the best-scoring one: selection runs on a
# 1,000-image slice of the test set, and the final checkpoint has been the
# stronger of the two wherever both were measured.
CKPT=$(ls -t checkpoints/"$RUN"/*/*/last.pt | head -1)
python -u -m scripts.benchmark -cfg "configs/$CFG.yaml" -cp "$CKPT" \
    2>&1 | tee "results/bench_${CFG}_final.txt"

echo "finished $(date)"
