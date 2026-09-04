#!/bin/bash
#SBATCH --job-name=q-bench-fn2
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=700G
#SBATCH --time=03:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
mkdir -p results

CKPT=$(ls -t checkpoints/mscoco-vqcfull-n2l3/*/*/last.pt | head -1)
echo "benchmarking $CKPT"
python -u -m scripts.benchmark -cfg configs/vqcfull_n2l3.yaml -cp "$CKPT" \
    2>&1 | tee results/bench_vqcfull_n2l3_last.txt
