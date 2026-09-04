"""Measure the spread of the loss over random initialisations, as a function of circuit width.

The standard barren-plateau signature is loss concentration at INITIALISATION:
draw random parameters many times, and if the spread of the loss shrinks
exponentially with circuit size, every starting point looks alike and there is
nothing for the optimiser to follow.

Earlier work here measured gradient norms on a TRAINED model, which cannot
answer this, a converged model sits in a basin by construction. This resamples
the text tower's parameters without any training and measures the variance of
the loss, for increasing qubits per grammatical type.

Only the text tower is re-initialised, so the quantity measured is the spread of
the loss as a function of the quantum parameters alone.

    python loss_concentration.py --samples 1000
    python loss_concentration.py --samples 10000 --captions 32
"""
from __future__ import annotations

import argparse
import pickle
import statistics
import time

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--captions", type=int, default=64)
    ap.add_argument("--qubits", default="1,2,3,4")
    ap.add_argument("--data", default="data/mscoco/processed/train_10k.pkl")
    args = ap.parse_args()

    from modules.compilation.quantum.ansatz import CustomV5Ansatz
    from modules.models.text.einsum_quantum import VQCModel
    from modules.models.fusion.criteria import FS_InfoNCE

    df = pickle.load(open(args.data, "rb"))
    diagrams = [d[0] for d in df["captions_diagram"][: args.captions]]
    bank = torch.load("data/mscoco/processed/image_embeddings.pt",
                      map_location="cpu")
    images = torch.stack([bank[int(i)].flatten().float()
                          for i in df["image_id"][: args.captions]])
    images = torch.nn.functional.normalize(images, dim=-1)

    loss_fn = FS_InfoNCE()
    rows = []

    for q in [int(x) for x in args.qubits.split(",")]:
        ansatz = CustomV5Ansatz(obmap={'n': q, 's': q, 'p': q, 'out': 9},
                                layers=2)
        # The ring keeps wide words affordable, which is what makes the higher
        # qubit counts measurable at all.
        recipes = [ansatz.tn2ansatz_ring(tn) for tn in diagrams]

        model = VQCModel(out_q=9)
        model.from_symbols([w for _, w in recipes], id_init=False)

        # Average circuit width over the sampled captions, which is the x axis
        # Tilen asked for.
        qubits = statistics.mean(
            sum(w['arity'] for w in words) for _, words in recipes)

        losses = []
        t = time.time()
        with torch.no_grad():
            for _ in range(args.samples):
                model.init_params(id_init=False)     # fresh random angles
                text = model(recipes)
                losses.append(loss_fn(text.flatten(1), images).item())

        var = statistics.pvariance(losses)
        rows.append((q, qubits, statistics.mean(losses), var))
        print(f"  n={q}: {qubits:6.1f} qubits per caption, "
              f"mean loss {rows[-1][2]:.4f}, variance {var:.3e}, "
              f"sd {var ** 0.5:.3e}   [{time.time() - t:.0f}s]")

    if len(rows) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot([r[1] for r in rows], [r[3] for r in rows], "o-",
                color="#2a78d6", lw=1.8, ms=6)
        for q, width, _, var in rows:
            ax.annotate(f"n={q}", (width, var), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9)
        ax.set_yscale("log")
        ax.set_xlabel("average qubits per caption", fontsize=10)
        ax.set_ylabel("variance of the loss at initialisation", fontsize=10)
        ax.set_title(f"Loss spread over {args.samples:,} random starts",
                     fontsize=11)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig("graphs/loss_concentration.png", dpi=150)
        print("\n  graphs/loss_concentration.png")

        first, last = rows[0], rows[-1]
        factor = first[3] / last[3] if last[3] > 0 else float("inf")
        print(f"  variance falls {factor:.1f}x from n={first[0]} to n={last[0]}, "
              f"while width grows {last[1] / first[1]:.1f}x")
        print("  A straight line on this log axis means exponential")
        print("  concentration, which is the barren-plateau signature.")


if __name__ == "__main__":
    main()
