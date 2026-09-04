#!/bin/bash
#SBATCH --job-name=qulip-coco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# Contraction is many tiny tensor ops driven from Python, so handing torch a
# large thread pool costs more in coordination than it saves. Leave the cores
# for parallel runs instead.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export QULIP_DEVICE=cpu
export QULIP_SKIP_METRICS=1
export BOBCAT_PARSER_PATH=$PWD/bobcat

source .venv/bin/activate
mkdir -p logs runs checkpoints

# Parsing all 591k COCO captions is the long pole and only needs doing once.
if [ ! -f data/mscoco/processed/train_full.pkl ]; then
    python -u -m scripts.retrieve --dataset mscoco --skip-images
    mv data/mscoco/processed/train.pkl data/mscoco/processed/train_full.pkl
fi

python -u -m scripts.train -cfg configs/coco_vqc_full.yaml
