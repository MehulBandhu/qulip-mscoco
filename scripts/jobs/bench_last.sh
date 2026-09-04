#!/bin/bash
#SBATCH --job-name=q-bench-last
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
mkdir -p results

# last.pt is written every epoch; best.pt only when validation improves, which
# on a 1,000-image slice can stall for several epochs while the model is still
# genuinely getting better.
CKPT=$(ls -t checkpoints/mscoco-vqcfull-fast/*/*/last.pt | head -1)
echo "benchmarking $CKPT"
python -u -m scripts.benchmark -cfg configs/vqcfull_fast.yaml -cp "$CKPT" \
    2>&1 | tee results/bench_vqcfull_fast_last.txt
