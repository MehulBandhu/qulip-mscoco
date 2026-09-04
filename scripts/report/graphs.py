"""Draw the figures from the training logs.

Reads logs/q-*.out and writes PNGs into graphs/. Nothing is retrained and no
checkpoints are touched, so it is safe to run while jobs are still going.

Where a config has been run more than once, the log with the most completed
epochs wins, and runs that collapsed to NaN are dropped.

    python -m scripts.graphs
"""
from __future__ import annotations

import re
from collections import defaultdict
from math import log
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no display on a compute node
import matplotlib.pyplot as plt

LOGS, OUT = Path("logs"), Path("graphs")

# Training images behind each config, for the scaling plots.
SIZES = {"5k": 5_000, "10k": 10_000, "20k": 20_000, "full": 118_287}

# Measured by pushing CLIP's own text encoder through this pipeline, so it is
# comparable to the model numbers rather than quoted from a paper.
CLIP_I2T = {1: 48.4, 5: 73.8, 10: 81.6}

EPOCH = re.compile(r"Epoch (\d+) \| Loss: ([\d.]+|nan)")
FIELD = re.compile(r"(\w+): (-?[\d.]+|nan)")
PARAMS = re.compile(r"Text=([\d,]+)")
TEST = re.compile(r"coco5k_(i2t|t2i)R(\d+)\s*\|\s*([\d.]+)")


def read_logs():
    """One record per config, preferring whichever run got furthest."""
    best = {}
    for path in sorted(LOGS.glob("q-*.out")):
        name = path.stem
        if name.startswith("q-bench"):
            continue
        tag = name.split("-")[1]
        text = path.read_text(errors="ignore").replace("\r", "\n")

        epochs = []
        for line in text.splitlines():
            m = EPOCH.search(line)
            if not m:
                continue
            rec = {"epoch": int(m.group(1))}
            for key, val in FIELD.findall(line.split("Loss:", 1)[1]):
                rec[key] = float("nan") if val == "nan" else float(val)
            rec["loss"] = float("nan") if m.group(2) == "nan" else float(m.group(2))
            epochs.append(rec)

        # A run that went NaN tells us nothing useful on these axes.
        epochs = [e for e in epochs if e["loss"] == e["loss"]]
        if not epochs:
            continue

        p = PARAMS.search(text)
        test = {f"{d}R{k}": float(v) * 100 for d, k, v in TEST.findall(text)}

        rec = {"tag": tag, "epochs": epochs, "test": test,
               "params": int(p.group(1).replace(",", "")) if p else None}
        if tag not in best or len(epochs) > len(best[tag]["epochs"]):
            best[tag] = rec
    return best


def series(rec, key):
    pts = [(e["epoch"], e[key]) for e in rec["epochs"] if key in e]
    return [p[0] for p in pts], [p[1] for p in pts]


def style(tag):
    """Quantum solid, classical dashed, colour by dataset size."""
    size = tag.replace("vqc", "").replace("tn", "")
    colours = {"5k": "#4C72B0", "10k": "#DD8452", "20k": "#55A868", "full": "#C44E52"}
    return colours.get(size, "#888"), ("-" if tag.startswith("vqc") else "--")


def label(tag):
    kind = "quantum" if tag.startswith("vqc") else "classical"
    size = tag.replace("vqc", "").replace("tn", "")
    return f"{kind}, {size}"


def main():
    OUT.mkdir(exist_ok=True)
    runs = read_logs()
    if not runs:
        raise SystemExit(f"no usable logs found in {LOGS}")
    print(f"{len(runs)} runs: {', '.join(sorted(runs))}\n")

    # --- training loss -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag in sorted(runs):
        c, ls = style(tag)
        ax.plot(*series(runs[tag], "loss"), color=c, ls=ls, lw=1.4, label=label(tag))
    # Chance differs by batch size: 256 for the quantum runs, 64 for classical.
    ax.axhline(log(256), color="#999", lw=0.8, ls=":")
    ax.axhline(log(64), color="#999", lw=0.8, ls=":")
    ax.text(0.5, log(256) + 0.06, "chance (batch 256)", color="#666", fontsize=7)
    ax.text(0.5, log(64) + 0.06, "chance (batch 64)", color="#666", fontsize=7)
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title("Training loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "loss.png", dpi=150); plt.close(fig)

    # --- recall against training time --------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, key in enumerate(["i2tR1", "i2tR10"]):
        for tag in sorted(runs):
            if not any(key in e for e in runs[tag]["epochs"]):
                continue
            c, ls = style(tag)
            x, y = series(runs[tag], key)
            ax[i].plot(x, [v * 100 for v in y], color=c, ls=ls, lw=1.4, label=label(tag))
        ax[i].set_xlabel("epoch"); ax[i].set_ylabel(f"{key} (%)")
        ax[i].grid(alpha=0.25); ax[i].legend(fontsize=8)
    ax[0].set_title("Recall@1 during training")
    ax[1].set_title("Recall@10 during training")
    fig.tight_layout(); fig.savefig(OUT / "recall_vs_epoch.png", dpi=150); plt.close(fig)

    # --- similarity gap ----------------------------------------------------
    # gamma is the mean similarity of true pairs minus the mean of everything
    # else, so it says whether the diagonal is pulling away from the rest.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for tag in sorted(runs):
        c, ls = style(tag)
        if any("gamma" in e for e in runs[tag]["epochs"]):
            ax[0].plot(*series(runs[tag], "gamma"), color=c, ls=ls, lw=1.4, label=label(tag))
        if any("i2t_hnm" in e for e in runs[tag]["epochs"]):
            ax[1].plot(*series(runs[tag], "i2t_hnm"), color=c, ls=ls, lw=1.4, label=label(tag))
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("mean true pair minus mean other")
    ax[0].set_title("Similarity gap")
    ax[1].axhline(0, color="#999", lw=0.8, ls=":")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("true pair minus hardest other")
    ax[1].set_title("Margin against the closest wrong caption")
    for a in ax:
        a.grid(alpha=0.25); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "similarity_gap.png", dpi=150); plt.close(fig)

    # --- recall against dataset size ---------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, marker in (("vqc", "o"), ("tn", "s")):
        for k in (1, 5, 10):
            pts = sorted(
                (SIZES[t.replace(kind, "")], r["test"].get(f"i2tR{k}"))
                for t, r in runs.items()
                if t.startswith(kind) and r["test"].get(f"i2tR{k}") is not None
            )
            if len(pts) < 2:
                continue
            name = "quantum" if kind == "vqc" else "classical"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker + "-",
                    lw=1.4, label=f"{name} R@{k}")
    for k, v in CLIP_I2T.items():
        ax.axhline(v, color="#999", lw=0.7, ls=":")
        ax.text(5200, v * 1.04, f"CLIP R@{k}", color="#666", fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training images"); ax.set_ylabel("image-to-text recall (%)")
    ax.set_title("Recall vs training set size")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "recall_vs_size.png", dpi=150); plt.close(fig)

    # --- model size --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, marker in (("vqc", "o"), ("tn", "s")):
        pts = sorted(
            (SIZES[t.replace(kind, "")], r["params"])
            for t, r in runs.items() if t.startswith(kind) and r["params"]
        )
        if pts:
            name = "quantum" if kind == "vqc" else "classical"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker + "-",
                    lw=1.6, label=name)
    ax.axhline(63_400_000, color="#999", lw=0.7, ls=":")
    ax.text(5200, 7.5e7, "CLIP text encoder", color="#666", fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training images"); ax.set_ylabel("text model parameters")
    ax.set_title("Model size vs training set size")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "model_size.png", dpi=150); plt.close(fig)

    # --- cost --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, marker in (("vqc", "o"), ("tn", "s")):
        pts = []
        for t, r in runs.items():
            if not t.startswith(kind):
                continue
            times = [e["Time"] for e in r["epochs"] if "Time" in e]
            if times:
                # Skip the first epoch: it pays a one-off contraction-path
                # search for every circuit shape it has not seen before.
                steady = times[1:] or times
                pts.append((SIZES[t.replace(kind, "")], sum(steady) / len(steady) / 60))
        if pts:
            pts.sort()
            name = "quantum" if kind == "vqc" else "classical"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker + "-",
                    lw=1.6, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training images"); ax.set_ylabel("minutes per epoch")
    ax.set_title("Time per epoch")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "time_per_epoch.png", dpi=150); plt.close(fig)

    for f in sorted(OUT.glob("*.png")):
        print(f"  {f}")
    print("\nGradient norms are not in the logs, so there is no figure for them.")


if __name__ == "__main__":
    main()
