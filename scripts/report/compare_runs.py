"""Figures comparing the two n=1 full-COCO runs.

They are the same model on the same data with the same seed; one runs the
gate-by-gate executor and one the compact word-first path. Plotting them
together is both the result and the proof that the rewrite changed nothing but
speed.

    python compare_runs.py

Writes into graphs/. Reads logs only, so it is safe while jobs are running.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS, OUT = Path("logs"), Path("graphs")

RUNS = {
    "gate-by-gate": "q-full-long-*.out",
    "word-first":   "q-full-fast-*.out",
}

# Test-set points measured so far, i2t R@1 / R@5 / R@10 on all 5,000 images.
TEST = {
    "gate-by-gate": {13: (1.90, 8.00, 13.38), 29: (2.62, 10.36, 17.12),
                     31: (2.64, 10.44, 17.18), 33: (2.76, 11.08, 17.94),
                     35: (2.82, 10.80, 17.92)},
}

LINE = re.compile(
    r"Epoch (\d+) \| Loss: ([\d.]+).*?i2tR1: ([\d.]+).*?i2tR10: ([\d.]+)"
    r".*?gamma: ([-\d.]+).*?(?:grad_norm: ([\d.e+-]+))?.*?Time: ([\d.]+)s")


def read(pattern):
    paths = sorted(LOGS.glob(pattern))
    if not paths:
        return []
    text = max(paths, key=lambda p: p.stat().st_size).read_text(
        errors="ignore").replace("\r", "\n")
    rows = []
    for line in text.splitlines():
        m = LINE.search(line)
        if m:
            ep, loss, r1, r10, gamma, grad, t = m.groups()
            rows.append({"epoch": int(ep), "loss": float(loss),
                         "r1": float(r1) * 100, "r10": float(r10) * 100,
                         "gamma": float(gamma),
                         "grad": float(grad) if grad else None,
                         "time": float(t)})
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    data = {k: read(v) for k, v in RUNS.items()}
    for k, v in data.items():
        print(f"  {k}: {len(v)} epochs")
    if not any(data.values()):
        raise SystemExit("no epoch lines found")

    colours = {"gate-by-gate": "#4C72B0", "word-first": "#C44E52"}

    # --- the two runs agree ------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for name, rows in data.items():
        if not rows:
            continue
        x = [r["epoch"] for r in rows]
        ax[0].plot(x, [r["loss"] for r in rows], color=colours[name], lw=1.6, label=name)
        ax[1].plot(x, [r["r1"] for r in rows], color=colours[name], lw=1.6, label=name)
        g = [(r["epoch"], r["grad"]) for r in rows if r["grad"] is not None]
        if g:
            ax[2].plot([p[0] for p in g], [p[1] for p in g],
                       color=colours[name], lw=1.6, label=name)

    ax[0].set_ylabel("training loss"); ax[0].set_title("Training loss")
    ax[1].set_ylabel("validation R@1 (%)"); ax[1].set_title("Validation recall")
    ax[2].set_ylabel("gradient norm"); ax[2].set_title("Gradient norm")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.25); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "two_executors.png", dpi=150); plt.close(fig)

    # --- how much faster ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, rows in data.items():
        if not rows:
            continue
        ax.plot([r["epoch"] for r in rows], [r["time"] / 60 for r in rows],
                color=colours[name], lw=1.6, label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("minutes per epoch")
    ax.set_title("Time per epoch"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "epoch_time.png", dpi=150); plt.close(fig)

    # --- test-set curve, with the bars worth clearing ----------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, pts in TEST.items():
        eps = sorted(pts)
        for i, (label, style) in enumerate([("R@1", "o-"), ("R@5", "s--"),
                                            ("R@10", "^:")]):
            ax.plot(eps, [pts[e][i] for e in eps], style, lw=1.5,
                    color=colours[name], alpha=1 - 0.25 * i, label=f"{name} {label}")
    # Tilen's reference points for tensor-network models.
    for y, txt in ((1.26, "best tensor network R@1"), (5.56, "best tensor network R@10")):
        ax.axhline(y, color="#888", lw=0.8, ls=":")
        ax.text(13.5, y * 1.05, txt, fontsize=7, color="#666")
    ax.set_xlabel("epoch"); ax.set_ylabel("test recall (%), 5,000 images")
    ax.set_title("Test set recall")
    ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "test_curve.png", dpi=150); plt.close(fig)

    for f in sorted(OUT.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
