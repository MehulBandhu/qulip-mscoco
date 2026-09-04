#!/bin/bash
#SBATCH --job-name=q-surface
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

for tag in "$@"; do
    dir=${tag//_/-}
    CKPT=$(ls -t checkpoints/mscoco-$dir/*/*/best.pt 2>/dev/null | head -1)
    [ -z "$CKPT" ] && { echo "=== $tag: no checkpoint"; continue; }
    echo "########## $tag  $CKPT"
    python -u -m scripts.loss_surface -cfg configs/$tag.yaml -cp "$CKPT" --grid 11
done
