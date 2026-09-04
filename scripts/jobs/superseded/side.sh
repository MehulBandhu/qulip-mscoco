#!/bin/bash
#SBATCH --job-name=q-side
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out
cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
CK=$(ls -t checkpoints/mscoco-vqcfull-fast/*/*/last.pt | head -1)
python -u which_side.py -cfg configs/vqcfull_fast.yaml -cp "$CK" \
    2>&1 | tee results/which_side.txt
