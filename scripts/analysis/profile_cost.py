"""Time each stage of a training batch: mapping, both towers, loss and optimiser.

The full-COCO run took 16.7 hours for its first epoch, but that epoch also pays
the one-off contraction-path search for every distinct circuit shape in the
dataset — around 107,000 of them. If that search is most of the cost, later
epochs are far cheaper and the run is viable. If it isn't, 16.7 hours is the
real price per epoch and the only way forward is splitting batches across cores.

This times the same batches twice in one process. The first pass builds the
cache, the second reuses it, so the gap between them is the search cost.

    python scripts/analysis/profile_cost.py --data data/mscoco/processed/train_full.pkl
"""
from __future__ import annotations

import argparse
import pickle
import time

from modules.compilation.quantum.ansatz import CustomV5Ansatz
from modules.models.text.einsum_quantum import VQCModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mscoco/processed/train_full.pkl")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=8)
    args = ap.parse_args()

    n = args.batch * args.batches
    print(f"loading {args.data}")
    df = pickle.load(open(args.data, "rb"))
    print(f"  {len(df)} images, using the first {n}")

    ansatz = CustomV5Ansatz(obmap={"n": 1, "s": 1, "p": 1, "out": 9}, layers=2)

    # One caption per image, matching what training samples each epoch.
    t0 = time.time()
    compiled, skipped = [], 0
    for diagrams in df["captions_diagram"][:n]:
        if not diagrams:
            skipped += 1
            continue
        compiled.append(ansatz(diagrams[0]))
    print(f"compiled {len(compiled)} circuits in {time.time() - t0:.1f}s "
          f"({skipped} images had no usable caption)")

    shapes = {expr for expr, _ in compiled}
    print(f"  {len(shapes)} distinct circuit shapes among them "
          f"({len(shapes) / len(compiled):.1%} unique)")

    model = VQCModel(out_q=9)
    t0 = time.time()
    model.from_symbols([arr for _, arr in compiled], id_init=True)
    print(f"from_symbols: {time.time() - t0:.1f}s, {len(model.symbols)} parameters")

    def sweep(label):
        times = []
        for i in range(0, len(compiled), args.batch):
            t = time.time()
            model(compiled[i : i + args.batch])
            times.append(time.time() - t)
            print(f"  {label} batch {i // args.batch:2}: {times[-1]:7.1f}s")
        cache = len(getattr(model, "path_cache", {}))
        print(f"  {label}: {sum(times):.1f}s total, cache holds {cache} paths")
        return sum(times)

    print("\ncold cache, first time seeing these shapes")
    cold = sweep("cold")

    print("\nwarm cache, identical batches")
    warm = sweep("warm")

    print(f"\ncold {cold:.1f}s | warm {warm:.1f}s | speedup {cold / warm:.2f}x")
    saved = (cold - warm) / cold if cold else 0
    print(f"path search accounts for {saved:.0%} of the first pass")

    per_epoch_cold = cold / len(compiled) * 118287
    per_epoch_warm = warm / len(compiled) * 118287
    print(f"\nscaled to a full-COCO epoch (118,287 images):")
    print(f"  first epoch  ~{per_epoch_cold / 3600:5.1f} hours")
    print(f"  later epochs ~{per_epoch_warm / 3600:5.1f} hours")

    if cold / warm > 2:
        print("\nMost of the cost is path search, so later epochs are much")
        print("cheaper and the full run is worth submitting as it stands.")
    else:
        print("\nThe cost is the contraction itself, not the search. Later epochs")
        print("won't get meaningfully faster; the run needs batches split across")
        print("cores to fit in any reasonable walltime.")


if __name__ == "__main__":
    main()
