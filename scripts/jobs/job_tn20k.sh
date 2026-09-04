#!/bin/bash
#SBATCH --job-name=q-tn20k
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export BOBCAT_PARSER_PATH=$PWD/bobcat QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1
# Few threads on purpose here: training is thousands of tiny contractions and a
# large pool costs more in coordination than it saves. The opposite of parsing.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "node $(hostname) | tn20k | $(date)"
python -u -m scripts.train -cfg configs/tn20k.yaml

CKPT=$(find checkpoints/mscoco-tn20k -name best.pt | head -1)
python -u -m scripts.benchmark -cfg configs/tn20k.yaml -cp "$CKPT"
echo "finished $(date)"
