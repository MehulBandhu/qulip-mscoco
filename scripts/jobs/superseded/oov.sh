#!/bin/bash
#SBATCH --job-name=q-oov
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out
cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
python -u oov_full.py 2>&1 | tee results/oov_full.txt
