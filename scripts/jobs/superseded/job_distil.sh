#!/bin/bash
#SBATCH --job-name=q-distil5k
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

python -u -m scripts.distil_words -cfg configs/vqc5k_spider.yaml --steps 300

CP=checkpoints/distilled/vqc5k_spider_distilled.pt

# warm start: distilled angles, then train everything
sed 's/mscoco-vqc5k-spider/mscoco-vqc5k-warm/' configs/vqc5k_spider.yaml \
    > configs/vqc5k_warm.yaml
python -u -m scripts.train -cfg configs/vqc5k_warm.yaml --init "$CP"
python -u -m scripts.benchmark -cfg configs/vqc5k_warm.yaml \
    -cp "$(ls -t checkpoints/mscoco-vqc5k-warm/*/*/best.pt | head -1)"

# frozen: distilled angles held fixed, only the image side learns
sed 's/mscoco-vqc5k-spider/mscoco-vqc5k-frozen/' configs/vqc5k_spider.yaml \
    > configs/vqc5k_frozen.yaml
python -u -m scripts.train -cfg configs/vqc5k_frozen.yaml --init "$CP" --freeze-text
python -u -m scripts.benchmark -cfg configs/vqc5k_frozen.yaml \
    -cp "$(ls -t checkpoints/mscoco-vqc5k-frozen/*/*/best.pt | head -1)"
