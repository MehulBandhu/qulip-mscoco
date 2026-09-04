#!/bin/bash
#SBATCH --job-name=q-budget
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

for tag in "$@"; do
    CKPT=$(ls -t checkpoints/mscoco-$tag/*/*/best.pt 2>/dev/null | head -1)
    [ -z "$CKPT" ] && { echo "=== $tag: no checkpoint"; continue; }
    echo "########## $tag  $CKPT"
    python -u -m scripts.param_budget -cfg configs/$tag.yaml -cp "$CKPT"
done
