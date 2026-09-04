#!/bin/bash
#SBATCH --job-name=q-bench
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

mkdir -p results
for tag in "$@"; do
    # Newest checkpoint directory, not the first one alphabetically — configs
    # get rerun and each run makes its own timestamped folder.
    dir=${tag//_/-}
    CKPT=$(ls -t checkpoints/mscoco-$dir/*/*/best.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then echo "=== $tag: no checkpoint"; continue; fi
    echo "=== $tag  $CKPT"
    python -u -m scripts.benchmark -cfg configs/$tag.yaml -cp "$CKPT" \
        2>&1 | tee results/bench_$tag.txt
done
