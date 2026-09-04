#!/bin/bash
#SBATCH --job-name=q-analyse
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

# Everything that needs a node: the two full test-set benchmarks, the parameter
# budget, and the loss surfaces. Read-only apart from writing into results/ and
# graphs/, so it will not disturb the training runs.

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
mkdir -p results graphs

bench () {
    run=$1; cfg=$2; which=$3
    ckpt=$(ls -t checkpoints/$run/*/*/$which 2>/dev/null | head -1)
    [ -z "$ckpt" ] && { echo "########## $run: no $which"; return; }
    echo "########## $run  $which  $ckpt"
    python -u -m scripts.benchmark -cfg configs/$cfg.yaml -cp "$ckpt" \
        2>&1 | tee results/bench_${run}_${which%.pt}.txt
}

# Full 5,000-image test set for both n=1 runs, from last.pt so the epoch is the
# latest rather than whichever one validation happened to like.
bench mscoco-vqcfull-long vqcfull_long last.pt
bench mscoco-vqcfull-fast vqcfull_fast last.pt

echo "########## parameter budget"
# The fast run saved init.pt before its first step, so drift has a real
# reference for the first time.
ck=$(ls -t checkpoints/mscoco-vqcfull-fast/*/*/last.pt | head -1)
python -u -m scripts.param_budget -cfg configs/vqcfull_fast.yaml -cp "$ck" \
    2>&1 | tee results/budget_vqcfull_fast.txt

echo "########## loss surfaces"
for pair in "mscoco-vqcfull-long vqcfull_long" "mscoco-vqcfull-fast vqcfull_fast"; do
    set -- $pair
    ck=$(ls -t checkpoints/$1/*/*/last.pt | head -1)
    [ -z "$ck" ] && continue
    echo "--- $2"
    python -u -m scripts.loss_surface -cfg configs/$2.yaml -cp "$ck" --grid 11
done

echo "finished $(date)"
