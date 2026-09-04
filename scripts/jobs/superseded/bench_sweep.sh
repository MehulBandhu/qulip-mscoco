#!/bin/bash
#SBATCH --job-name=q-bench-v
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
mkdir -p results

for v in 2 4 8 16; do
    ck=$(ls -t checkpoints/mscoco-vqc10k-v$v/*/*/best.pt 2>/dev/null | head -1)
    [ -z "$ck" ] && { echo "########## v$v: no checkpoint"; continue; }
    echo "########## v$v  $ck"
    python -u -m scripts.benchmark -cfg configs/vqc10k_v$v.yaml -cp "$ck" \
        2>&1 | tee results/bench_vqc10k_v$v.txt
done
