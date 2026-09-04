"""Figures across all the full-COCO runs.

Reads training logs and saved benchmark files, so it can run at any time and
picks up whatever has been measured. Nothing is recomputed.

    python figures2.py

Writes PNGs into graphs/.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS, RESULTS, OUT = Path("logs"), Path("results"), Path("graphs")

# label -> (log pattern, colour). The first is the completed baseline.
RUNS = {
    "1 qubit per type":  ("q-full-fast-*.out",   "#2a78d6"),
    "2 qubits per type": ("q-full-fastn2-*.out", "#c0392b"),
    "3 qubits per type": ("q-full-n3r-*.out",    "#1baf7a"),
    "2 captions/image":  ("q-full-k2-*.out",     "#e08a1e"),
    "5 captions/image":  ("q-full-k5-*.out",     "#7d4bb5"),
}

# Test-set measurements, i2t R@1/R@5/R@10 on all 5,000 images.
TEST = {
    "1 qubit per type": {13: (1.90, 8.00, 13.38), 29: (2.62, 10.36, 17.12),
                         33: (2.76, 11.08, 17.94), 42: (2.82, 10.86, 18.56),
                         91: (3.12, 11.96, 19.20), 99: (3.08, 12.46, 20.10)},
    "2 qubits per type": {12: (2.14, 9.10, 14.92), 24: (2.68, 10.88, 18.66),
                          32: (2.92, 12.28, 19.52), 37: (2.94, 11.92, 19.64)},
    "2 captions/image": {39: (3.00, 10.78, 18.02), 46: (2.78, 11.06, 18.46)},
    "5 captions/image": {15: (2.72, 10.06, 16.96)},
    "3 qubits per type": {4: (0.98, 4.78, 8.42)},
}

LINE = re.compile(
    r"Epoch (\d+) \| Loss: ([\d.]+).*?i2tR1: ([\d.]+).*?i2tR10: ([\d.]+)"
    r".*?gamma: ([-\d.]+).*?grad_norm: ([\d.e+-]+)")


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
            e, loss, r1, r10, gamma, grad = m.groups()
            rows.append(dict(epoch=int(e), loss=float(loss), r1=float(r1) * 100,
                             r10=float(r10) * 100, gamma=float(gamma),
                             grad=float(grad)))
    return rows


def style(ax, x, y, title):
    ax.set_xlabel(x, fontsize=10); ax.set_ylabel(y, fontsize=10)
    ax.set_title(title, fontsize=11); ax.grid(alpha=0.25)
    ax.tick_params(labelsize=9)


def main():
    OUT.mkdir(exist_ok=True)
    data = {k: read(v[0]) for k, v in RUNS.items()}
    for k, v in data.items():
        print(f"  {k}: {len(v)} epochs")
    live = {k: v for k, v in data.items() if v}
    if not live:
        raise SystemExit("no epoch lines found")

    # 1. training loss ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, rows in live.items():
        ax.plot([r["epoch"] for r in rows], [r["loss"] for r in rows],
                color=RUNS[name][1], lw=1.7, label=name)
    style(ax, "epoch", "training loss", "Training loss")
    ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "10_loss.png", dpi=150); plt.close(fig)

    # 2. validation ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for name, rows in live.items():
        x = [r["epoch"] for r in rows]
        ax[0].plot(x, [r["r1"] for r in rows], color=RUNS[name][1], lw=1.5,
                   label=name)
        ax[1].plot(x, [r["r10"] for r in rows], color=RUNS[name][1], lw=1.5,
                   label=name)
    style(ax[0], "epoch", "correct caption found (%)",
          "Validation, top 1 (1,000 images)")
    style(ax[1], "epoch", "correct caption found (%)",
          "Validation, top 10 (1,000 images)")
    ax[0].legend(fontsize=8.5); ax[1].legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "11_validation.png", dpi=150); plt.close(fig)

    # 3. test set -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for name, pts in TEST.items():
        if not pts:
            continue
        eps = sorted(pts)
        c = RUNS[name][1]
        ax[0].plot(eps, [pts[e][0] for e in eps], "o-", color=c, lw=1.6, ms=5,
                   label=name)
        ax[1].plot(eps, [pts[e][2] for e in eps], "o-", color=c, lw=1.6, ms=5,
                   label=name)
    for a, y, txt in ((ax[0], 1.26, "best tensor-network result"),
                      (ax[1], 5.56, "best tensor-network result")):
        a.axhline(y, color="#8a8a8a", lw=0.9, ls=":")
        a.text(5, y * 1.08, txt, fontsize=8, color="#666")
    for a, y, txt in ((ax[0], 1.72, "classical model"),
                      (ax[1], 9.84, "classical model")):
        a.axhline(y, color="#8a8a8a", lw=0.9, ls="--")
        a.text(5, y * 1.05, txt, fontsize=8, color="#666")
    style(ax[0], "epoch", "correct caption found (%)",
          "Test, top 1 (5,000 images)")
    style(ax[1], "epoch", "correct caption found (%)",
          "Test, top 10 (5,000 images)")
    ax[0].legend(fontsize=8.5); ax[1].legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "12_test.png", dpi=150); plt.close(fig)

    # 4. what the model is learning ----------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for name, rows in live.items():
        x = [r["epoch"] for r in rows]
        ax[0].plot(x, [r["gamma"] for r in rows], color=RUNS[name][1], lw=1.5,
                   label=name)
        ax[1].plot(x, [r["grad"] for r in rows], color=RUNS[name][1], lw=1.5,
                   label=name)
    style(ax[0], "epoch", "similarity gap",
          "How far true pairs sit above the rest")
    style(ax[1], "epoch", "gradient norm", "Gradient norm")
    ax[0].legend(fontsize=8.5); ax[1].legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "13_learning.png", dpi=150); plt.close(fig)

    for f in sorted(OUT.glob("1*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
