"""Save the per-pair margins from an ARO evaluation.

The accuracy already reported is just the fraction of pairs where the margin is
positive, which cannot distinguish two very different failures. A model that
cannot tell the orderings apart has margins piled around zero; a model that has
learned the wrong ordering has them shifted negative. The bag-of-words variant
sits 23 to 25 standard errors below chance, and that is worth resolving.

Writes report/aro_margins_<tag>.npz with the positive and negative similarity
for every pair, plus a short summary.

    python aro_margins.py -cfg configs/aro_vqcfull_fast.yaml -cp <checkpoint> --tag n2l5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from modules.data_pipeline.engine import DataEngine
from modules.utils.factory import build_experiment
from modules.utils.general import CheckpointManager, setup_exp

REPORT = Path("report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-cfg", "--config", required=True)
    ap.add_argument("-cp", "--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", default="att,rel")
    args = ap.parse_args()

    config, device, _ = setup_exp(args.config)
    ansatz, image_model, text_model, _ = build_experiment(config, device)
    engine = DataEngine(config, ansatz, device)
    engine.text_init(text_model, pd.concat(
        [engine.compile_text("train"), engine.compile_text("val")],
        ignore_index=True))
    CheckpointManager.load_model(args.checkpoint, image_model, text_model, device)
    text_model.eval(); image_model.eval()

    out = {}
    for split in args.splits.split(","):
        compiled = engine.compile_text(split)
        loader = engine.get_loader(compiled, split=split)
        pos, neg = [], []
        with torch.no_grad():
            for batch in loader:
                img = F.normalize(
                    image_model(batch["image"].to(device)).flatten(1), dim=1)
                p = F.normalize(
                    text_model(batch["pos_caption"]).flatten(1), dim=1)
                n = F.normalize(
                    text_model(batch["neg_caption"]).flatten(1), dim=1)
                # Same quantity the reported accuracy thresholds at zero.
                pos.extend(torch.sum(img.conj() * p, dim=1).abs().cpu().numpy())
                neg.extend(torch.sum(img.conj() * n, dim=1).abs().cpu().numpy())

        pos, neg = np.array(pos), np.array(neg)
        margin = pos - neg
        out[f"{split}_pos"], out[f"{split}_neg"] = pos, neg

        acc = (margin > 0).mean()
        se = np.sqrt(0.25 / len(margin))
        print(f"\n  {split}: {len(margin):,} pairs, accuracy {acc:.4f} "
              f"({(acc - 0.5) / se:+.1f} SE from chance)")
        print(f"    margin  mean {margin.mean():+.5f}  "
              f"median {np.median(margin):+.5f}  sd {margin.std():.5f}")
        print(f"    as a fraction of its own spread: "
              f"{margin.mean() / margin.std():+.3f}")
        print(f"    positive similarity {pos.mean():.4f}, "
              f"negative {neg.mean():.4f}")
        # A shift this large relative to the spread means the model prefers the
        # corrupted caption systematically, not that it cannot tell them apart.
        if abs(margin.mean()) > 0.15 * margin.std():
            print("    the distribution is shifted, not centred: the model "
                  "prefers one side systematically")
        else:
            print("    the distribution is centred near zero: the model is "
                  "not distinguishing the two orderings")

    REPORT.mkdir(exist_ok=True)
    np.savez(REPORT / f"aro_margins_{args.tag}.npz", **out)
    print(f"\n  report/aro_margins_{args.tag}.npz")


if __name__ == "__main__":
    main()
