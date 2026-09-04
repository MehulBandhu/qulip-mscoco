"""Map the loss landscape around a trained checkpoint.

Follows Li et al. (2018): take the trained parameters, pick two random
directions, normalise them to the scale of the weights themselves, and sweep a
grid of loss values across the plane they span.

Worth doing for this project because the usual explanation for why variational
circuits train badly is barren plateaus - landscapes flat enough that gradients
carry almost no signal. Running this on both towers says whether that is what
is happening here or whether it is something else.

    python scripts/analysis/loss_surface.py -cfg configs/vqc5k.yaml -cp <best.pt> --grid 11

Writes graphs/surface_<name>.png and the raw grid alongside it as .npz, so the
plot can be redrawn without recomputing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.data_pipeline.engine import DataEngine
from modules.utils.factory import build_experiment
from modules.utils.general import CheckpointManager, setup_exp


def random_direction(theta: torch.Tensor) -> torch.Tensor:
    """A random direction scaled to the size of the weights it perturbs.

    Without the rescaling, a fixed step means something quite different for a
    model with 250k parameters than one with 2 billion, and the two surfaces
    would not be comparable.
    """
    d = torch.randn_like(theta)
    return d * (theta.norm() / d.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-cfg", "--config", required=True)
    ap.add_argument("-cp", "--checkpoint", required=True)
    ap.add_argument("--grid", type=int, default=11, help="points per axis")
    ap.add_argument("--span", type=float, default=1.0, help="half-width in units of |theta|")
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config, device, _ = setup_exp(args.config)
    ansatz, image_model, text_model, loss_fn = build_experiment(config, device)

    engine = DataEngine(config, ansatz, device)
    train = engine.compile_text("train")
    val = engine.compile_text("val")
    import pandas as pd
    engine.text_init(text_model, pd.concat([train, val], ignore_index=True))
    CheckpointManager.load_model(args.checkpoint, image_model, text_model, device)

    loader = engine.get_loader(train, split="train")
    batches = []
    for i, batch in enumerate(loader):
        batches.append(engine.eval_mapper(batch))
        if i + 1 >= args.batches:
            break
    print(f"holding {len(batches)} batch(es) fixed for every grid point")

    text_model.eval()
    image_model.eval()

    theta = parameters_to_vector(text_model.parameters()).detach().clone()
    torch.manual_seed(args.seed)
    d1, d2 = random_direction(theta), random_direction(theta)
    print(f"{theta.numel():,} parameters, |theta| = {theta.norm():.3f}")

    def loss_at(alpha, beta):
        vector_to_parameters(theta + alpha * d1 + beta * d2, text_model.parameters())
        total = 0.0
        with torch.no_grad():
            for images, texts in batches:
                total += loss_fn(text_model(texts), image_model(images)).item()
        return total / len(batches)

    coords = np.linspace(-args.span, args.span, args.grid)
    surface = np.zeros((args.grid, args.grid))
    for i, a in enumerate(coords):
        for j, b in enumerate(coords):
            surface[i, j] = loss_at(float(a), float(b))
        print(f"  row {i + 1}/{args.grid}  min {surface[i].min():.3f}  "
              f"max {surface[i].max():.3f}")

    vector_to_parameters(theta, text_model.parameters())   # put it back

    name = Path(args.config).stem
    out = Path("graphs")
    out.mkdir(exist_ok=True)
    np.savez(out / f"surface_{name}.npz", surface=surface, coords=coords)

    centre = surface[args.grid // 2, args.grid // 2]
    print(f"\nloss at the trained point: {centre:.4f}")
    print(f"across the sweep: {surface.min():.4f} to {surface.max():.4f}")
    print(f"range as a fraction of the centre: {(surface.max() - surface.min()) / centre:.3f}")
    print("A small fraction means a flat landscape, which is the barren-plateau"
          " signature.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    cs = ax[0].contourf(coords, coords, surface, levels=25, cmap="viridis")
    ax[0].contour(coords, coords, surface, levels=12, colors="white", linewidths=0.4)
    ax[0].plot(0, 0, "r+", markersize=10)
    fig.colorbar(cs, ax=ax[0], label="loss")
    ax[0].set_xlabel("direction 1"); ax[0].set_ylabel("direction 2")
    ax[0].set_title(f"Loss surface, {name}")

    mid = args.grid // 2
    ax[1].plot(coords, surface[mid, :], label="along direction 1")
    ax[1].plot(coords, surface[:, mid], label="along direction 2")
    ax[1].axvline(0, color="#999", lw=0.8, ls=":")
    ax[1].set_xlabel("distance from the trained point")
    ax[1].set_ylabel("loss")
    ax[1].set_title("Slices through the centre")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out / f"surface_{name}.png", dpi=150)
    print(f"\nwrote graphs/surface_{name}.png")


if __name__ == "__main__":
    main()
