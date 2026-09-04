#!/bin/bash
#SBATCH --job-name=q-tn10kseed
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

python -u -m scripts.seed_clip -cfg configs/tn10k.yaml

sed 's/mscoco-tn10k/mscoco-tn10k-seeded/' configs/tn10k.yaml > configs/tn10k_seeded.yaml
python -u -m scripts.train -cfg configs/tn10k_seeded.yaml \
    --init checkpoints/seeded/tn10k_seeded.pt

CKPT=$(ls -t checkpoints/mscoco-tn10k-seeded/*/*/best.pt | head -1)
python -u -m scripts.benchmark -cfg configs/tn10k_seeded.yaml -cp "$CKPT"
