#!/bin/bash
#SBATCH --job-name=q-vqc10kseed
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

python -u -m scripts.train -cfg configs/vqc10k_seeded.yaml \
    --init checkpoints/seeded/vqc10k_angles.pt

CKPT=$(ls -t checkpoints/mscoco-vqc10k-seeded/*/*/best.pt | head -1)
python -u -m scripts.benchmark -cfg configs/vqc10k_seeded.yaml -cp "$CKPT"
