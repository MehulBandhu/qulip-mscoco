"""Compare a caption's state with that of its word-order-corrupted variant.

ARO transfer inverted as retrieval improved, 0.563 on relation at epoch 13,
0.410 by epoch 99, both against chance 0.50. Two things worth separating.

First, attribution against relation: they are different corruptions (swapped
adjectives versus swapped subject and object), so a model might handle one and
not the other, and lumping them hides that.

Second, and more diagnostic: the cosine similarity between a caption's state and
its corrupted variant. ARO pairs differ only in word order, so if the two states
sit almost on top of each other the model simply cannot represent the
difference, whatever the accuracy says. That distinguishes "represents word
order but ranks it wrong" from "cannot see word order at all".

    python permutation_sim.py -cfg configs/aro_vqcfull_fast.yaml -cp <checkpoint>
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from modules.data_pipeline.engine import DataEngine
from modules.utils.factory import build_experiment
from modules.utils.general import CheckpointManager, setup_exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-cfg", "--config", required=True)
    ap.add_argument("-cp", "--checkpoint", required=True)
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

    for split in args.splits.split(","):
        compiled = engine.compile_text(split)
        t_e = next(c for c in compiled.columns if c.startswith("true_caption")
                   and c.endswith("_einsum"))
        t_s = t_e.replace("_einsum", "_symbols")
        f_e = next(c for c in compiled.columns if c.startswith("false_caption")
                   and c.endswith("_einsum"))
        f_s = f_e.replace("_einsum", "_symbols")

        sims, correct, n = [], 0, 0
        with torch.no_grad():
            for s in range(0, len(compiled), 128):
                chunk = compiled.iloc[s:s + 128]
                pos = text_model(list(zip(chunk[t_e], chunk[t_s]))).flatten(1)
                neg = text_model(list(zip(chunk[f_e], chunk[f_s]))).flatten(1)
                img = torch.stack([
                    engine.dataset._load_image(i).flatten()
                    if hasattr(engine, "dataset") else torch.zeros(1)
                    for i in chunk["image_id"]]) if False else None

                a = F.normalize(pos.to(torch.complex64), dim=-1)
                b = F.normalize(neg.to(torch.complex64), dim=-1)
                sims.extend((a * b.conj()).sum(-1).abs().tolist())
                n += len(chunk)

        sims = torch.tensor(sims)
        print(f"\n  {split}: {n:,} pairs")
        print(f"    cosine similarity between a caption and its corrupted twin")
        print(f"      mean {sims.mean():.4f}   median {sims.median():.4f}")
        print(f"      10th {sims.quantile(0.1):.4f}   90th {sims.quantile(0.9):.4f}")
        print(f"      above 0.99: {(sims > 0.99).float().mean():.1%}   "
              f"above 0.95: {(sims > 0.95).float().mean():.1%}")

    print("\n  A mean near 1.0 means the two orderings produce almost the same")
    print("  state, so the model cannot represent the difference at all and the")
    print("  accuracy is decided by noise. Well below 1.0 means it does")
    print("  distinguish them and is simply ranking them the wrong way round.")


if __name__ == "__main__":
    main()
