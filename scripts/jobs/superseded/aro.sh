#!/bin/bash
#SBATCH --job-name=q-aro
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
mkdir -p results

# The ARO configs keep COCO for train and val so the model keeps the vocabulary
# it learned, and only the test split moves. Checkpoint directories are named
# after the original COCO runs, so they are paired explicitly here.
run () {
    cfg=$1; dir=$2
    ckpt=$(ls -t checkpoints/$dir/*/*/best.pt 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then echo "########## $cfg: no checkpoint under $dir"; return; fi
    echo "########## $cfg  $ckpt"
    python -u -m scripts.benchmark -cfg configs/$cfg.yaml -cp "$ckpt" \
        2>&1 | tee results/$cfg.txt
}

run aro_vqc20k        mscoco-vqc20k
run aro_vqc20k_spider mscoco-vqc20k-spider
run aro_vqcfull       mscoco-vqcfull
