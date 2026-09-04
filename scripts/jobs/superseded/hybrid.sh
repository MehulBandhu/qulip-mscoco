#!/bin/bash
#SBATCH --job-name=q-hybrid
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

G=$(ls -t checkpoints/mscoco-vqc20k/*/*/best.pt | head -1)
L=$(ls -t checkpoints/mscoco-vqc20k-spider/*/*/best.pt | head -1)
echo "grammar $G"
echo "lexical $L"

echo "########## COCO"
python -u -m scripts.hybrid_score --grammar configs/vqc20k.yaml --grammar-cp "$G" \
    --lexical configs/vqc20k_spider.yaml --lexical-cp "$L" --split coco5k

for split in att rel; do
    echo "########## ARO $split"
    python -u -m scripts.hybrid_score --grammar configs/aro_vqc20k.yaml --grammar-cp "$G" \
        --lexical configs/aro_vqc20k_spider.yaml --lexical-cp "$L" --split $split
done
