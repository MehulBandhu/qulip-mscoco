#!/bin/bash
#SBATCH --job-name=q-bench-grid
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=700G
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

# Every run at whatever epoch it has reached, on the full 5,000-image test set.
# 700 GB because bond dimension is 2^L, so cores at L=4 are four times the size
# of L=2 and the test pass encodes all 24,909 captions at once - 128 GB was
# enough for L=2 and not for L=3.
#
# last.pt throughout: the 1,000-image validation slice picks checkpoints badly,
# and on the converged n=1 run the final checkpoint beat the "best" one on five
# of six metrics.

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
mkdir -p results

run () {
    tag=$1; cfg=$2
    ck=$(ls -t checkpoints/mscoco-vqcfull-$tag/*/*/last.pt 2>/dev/null | head -1)
    if [ -z "$ck" ]; then echo "########## $tag: no checkpoint"; return; fi
    ep=$(basename $(dirname "$ck"))
    echo "########## $tag  $ck"
    python -u -m scripts.benchmark -cfg configs/$cfg.yaml -cp "$ck" \
        2>&1 | tee results/bench_${tag}_grid.txt
}

# Leaders first, so a failure later still leaves the numbers that matter.
run n2l5    vqcfull_n2l5
run n2l6    vqcfull_n2l6
run n3r2    vqcfull_n3r2
run n2l4q10    vqcfull_n2l4q10
run n2l4q11    vqcfull_n2l4q11
run n2l3    vqcfull_n2l3
run n3ring  vqcfull_n3ring
run n3l4    vqcfull_n3l4
run n2l4    vqcfull_n2l4
run n3l3    vqcfull_n3l3
run n4ring  vqcfull_n4ring
run fastn2  vqcfull_fastn2
run k5      vqcfull_k5

echo
echo "##### reference, i2t R@1 / R@5 / R@10 on the same 5,000 images"
echo "#####   n=1 converged ep99      3.08 / 12.46 / 20.10"
echo "#####   K=2 converged ep97      3.52 / 12.16 / 20.06"
echo "#####   classical grammar       1.72 /  6.30 /  9.84"
echo "#####   best tensor network     1.26 /    -  /  5.56"
echo "##### finished $(date)"
