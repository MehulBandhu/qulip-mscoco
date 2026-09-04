#!/bin/bash
#SBATCH --job-name=q-bench-all
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

# Every full-COCO run at whatever epoch it has reached, on the full 5,000-image
# test set. Uses last.pt rather than best.pt: the 1,000-image validation slice
# picks checkpoints badly, and on the converged n=1 run the final checkpoint beat
# the "best" one on five of six metrics.

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
mkdir -p results

run () {
    tag=$1; cfg=$2
    ck=$(ls -t checkpoints/mscoco-vqcfull-$tag/*/*/last.pt 2>/dev/null | head -1)
    if [ -z "$ck" ]; then echo "########## $tag: no checkpoint"; return; fi
    echo "########## $tag  $ck"
    python -u -m scripts.benchmark -cfg configs/$cfg.yaml -cp "$ck" \
        2>&1 | tee results/bench_${tag}_all.txt
}

# Cheapest first, so the slow ones cannot starve the rest.
run k2      vqcfull_k2
run k5      vqcfull_k5
run n3ring  vqcfull_n3ring
run fastn2  vqcfull_fastn2
run n2l3    vqcfull_n2l3
run n2      vqcfull_n2

echo
echo "##### reference points, i2t R@1 / R@5 / R@10 on the same 5,000 images"
echo "#####   n=1 converged, epoch 99   3.08 / 12.46 / 20.10   sum 63.17"
echo "#####   n=1 epoch 42              2.82 / 10.86 / 18.56"
echo "#####   classical grammar         1.72 /  6.30 /  9.84"
echo "#####   best tensor network       1.26 /    -  /  5.56"
echo "##### finished $(date)"
