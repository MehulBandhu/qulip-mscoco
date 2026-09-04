#!/bin/bash
#SBATCH --job-name=q-bench-fn2
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
mkdir -p results

CKPT=$(ls -t checkpoints/mscoco-vqcfull-k2/*/*/last.pt | head -1)
echo "benchmarking $CKPT"
python -u -m scripts.benchmark -cfg configs/vqcfull_k2.yaml -cp "$CKPT" \
    2>&1 | tee results/bench_vqcfull_k2_last.txt
