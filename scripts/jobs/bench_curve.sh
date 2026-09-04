#!/bin/bash
#SBATCH --job-name=q-curve
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --output=/cephfs/mbandhu/qulip/logs/%x-%j.out
cd /cephfs/mbandhu/qulip
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4

# Test-set recall at every saved snapshot, so the curve is measured on the same
# 5,000 images the headline numbers use rather than the 1,000-image slice.
for pair in "n3l4:vqcfull_n3l4" "n2l4:vqcfull_n2l4" "fast:vqcfull_fast"; do
    run=${pair%%:*}; cfg=${pair##*:}
    d=$(ls -dt checkpoints/mscoco-vqcfull-$run/*/*/ | head -1)
    for ck in $(ls "$d"params_ep*.pt 2>/dev/null) "$d"last.pt; do
        [ -f "$ck" ] || continue
        echo "########## $run $(basename $ck)"
        python -u -m scripts.benchmark -cfg configs/$cfg.yaml -cp "$ck" \
            2>&1 | grep -E "Recovered|coco5k_i2tR"
    done
done
