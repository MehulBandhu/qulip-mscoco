#!/bin/bash
#SBATCH --job-name=q-tn5k
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export BOBCAT_PARSER_PATH=$PWD/bobcat QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1
# Few threads on purpose here: training is thousands of tiny contractions and a
# large pool costs more in coordination than it saves. The opposite of parsing.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "node $(hostname) | tn5k | $(date)"
python -u -m scripts.train -cfg configs/tn5k.yaml

CKPT=$(find checkpoints/mscoco-tn5k -name best.pt | head -1)
python -u -m scripts.benchmark -cfg configs/tn5k.yaml -cp "$CKPT"
echo "finished $(date)"
