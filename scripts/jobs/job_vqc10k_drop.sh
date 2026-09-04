#!/bin/bash
#SBATCH --job-name=q-vqc10k_drop
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export BOBCAT_PARSER_PATH=$PWD/bobcat QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1
# Few threads on purpose here: training is thousands of tiny contractions and a
# large pool costs more in coordination than it saves. The opposite of parsing.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "node $(hostname) | vqc10k_drop | $(date)"
python -u -m scripts.train -cfg configs/vqc10k_drop.yaml

CKPT=$(ls -t checkpoints/mscoco-vqc10k-drop/*/*/best.pt 2>/dev/null | head -1)
python -u -m scripts.benchmark -cfg configs/vqc10k_drop.yaml -cp "$CKPT"
echo "finished $(date)"
