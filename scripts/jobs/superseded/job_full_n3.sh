#!/bin/bash
#SBATCH --job-name=q-full-n3
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=700G
#SBATCH --time=10-00:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

echo "node $(hostname) | started $(date)"
python -u -m scripts.train -cfg configs/vqcfull_n3.yaml || exit 1

CKPT=$(ls -t checkpoints/mscoco-vqcfull-n3/*/*/best.pt | head -1)
python -u -m scripts.benchmark -cfg configs/vqcfull_n3.yaml -cp "$CKPT"
echo "finished $(date)"
