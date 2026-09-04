"""Draw the saved loss grid as a 3D surface.

loss_surface.py writes the raw grid alongside its contour plot, so this reads
that and redraws it without recomputing anything.

    python scripts/analysis/surface3d.py                       # every grid it finds
    python scripts/analysis/surface3d.py --name vqcfull        # just one
    python scripts/analysis/surface3d.py --elev 40 --azim -120 # change the viewpoint
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("graphs")


def draw(path: Path, elev: float, azim: float):
    data = np.load(path)
    surface, coords = data["surface"], data["coords"]
    name = path.stem.replace("surface_", "")

    x, y = np.meshgrid(coords, coords)
    mid = len(coords) // 2
    centre = surface[mid, mid]

    fig = plt.figure(figsize=(11, 4.5))

    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(x, y, surface, cmap="viridis", linewidth=0,
                    antialiased=True, alpha=0.9)
    # The trained parameters sit at the origin of both directions.
    ax.scatter([0], [0], [centre], color="red", s=40, depthshade=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("direction 1", fontsize=9)
    ax.set_ylabel("direction 2", fontsize=9)
    ax.set_zlabel("loss", fontsize=9)
    ax.set_title(f"Loss surface, {name}", fontsize=11)
    ax.tick_params(labelsize=8)

    ax2 = fig.add_subplot(122)
    cs = ax2.contourf(coords, coords, surface, levels=25, cmap="viridis")
    ax2.contour(coords, coords, surface, levels=12, colors="white", linewidths=0.4)
    ax2.plot(0, 0, "r+", markersize=12, markeredgewidth=2)
    fig.colorbar(cs, ax=ax2, label="loss")
    ax2.set_xlabel("direction 1"); ax2.set_ylabel("direction 2")
    ax2.set_title("Seen from above", fontsize=11)

    fig.tight_layout()
    target = OUT / f"surface3d_{name}.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)

    span = (surface.max() - surface.min()) / centre
    print(f"  {target}")
    print(f"    loss at the trained point {centre:.4f}, "
          f"range {surface.min():.4f} to {surface.max():.4f} ({span:.3f} of centre)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="one grid, by config name")
    ap.add_argument("--elev", type=float, default=35)
    ap.add_argument("--azim", type=float, default=-135)
    args = ap.parse_args()

    pattern = f"surface_{args.name}.npz" if args.name else "surface_*.npz"
    grids = sorted(OUT.glob(pattern))
    if not grids:
        raise SystemExit(f"no grids matching {OUT}/{pattern} run loss_surface first")

    for path in grids:
        draw(path, args.elev, args.azim)


if __name__ == "__main__":
    main()
