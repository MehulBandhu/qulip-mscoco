#!/bin/bash
#SBATCH --job-name=b-errors
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --exclude=worker086
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out
cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

# Best and baseline, so the failure pattern can be compared across capacity.
for pair in "n3l4:vqcfull_n3l4" "fast:vqcfull_fast"; do
    run=${pair%%:*}; cfg=${pair##*:}
    ck=$(ls -t checkpoints/mscoco-vqcfull-$run/*/*/last.pt | head -1)
    python -u error_analysis.py -cfg configs/$cfg.yaml -cp "$ck" --tag $run
done
