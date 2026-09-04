#!/bin/bash
#SBATCH --job-name=q-aro-fast
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out
cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
python -u -m scripts.benchmark -cfg configs/aro_vqcfull_fast.yaml -cp "checkpoints/mscoco-vqcfull-fast/vqc/t_lay2_n1_p1_s1__v_mlp_lay4__0821_2300/last.pt"     2>&1 | tee results/aro_vqcfull_fast.txt
