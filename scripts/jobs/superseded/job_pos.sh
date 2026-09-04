#!/bin/bash
#SBATCH --job-name=q-pos10k
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

python -u -m scripts.train -cfg configs/vqc10k_pos.yaml
CKPT=$(ls -t checkpoints/mscoco-vqc10k-pos/*/*/best.pt | head -1)

echo "########## COCO retrieval"
python -u -m scripts.benchmark -cfg configs/vqc10k_pos.yaml -cp "$CKPT"

echo "########## ARO transfer, chance 0.50"
python -u -m scripts.benchmark -cfg configs/vqc10k_pos_aro.yaml -cp "$CKPT"
