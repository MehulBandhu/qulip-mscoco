#!/bin/bash
#SBATCH --job-name=qulip-parse
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out

cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export BOBCAT_PARSER_PATH=$PWD/bobcat
export QULIP_DEVICE=cpu

# Parsing is BERT inference and wants every core. Training is the opposite:
# thousands of tiny contractions where a large thread pool costs more in
# coordination than it saves. Different steps, different settings.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "node $(hostname) | threads $OMP_NUM_THREADS | started $(date)"
python -u -m scripts.retrieve --dataset mscoco --skip-images
mv data/mscoco/processed/train.pkl data/mscoco/processed/train_full.pkl
echo "finished $(date)"
