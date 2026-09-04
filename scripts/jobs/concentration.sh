#!/bin/bash
#SBATCH --job-name=q-conc
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out
cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu OMP_NUM_THREADS=4
mkdir -p results graphs
python -u loss_concentration.py --samples 10000 --captions 64 \
    --qubits 1,2,3,4,5 2>&1 | tee results/loss_concentration.txt
