"""Harvest every run into structured tables.

Reads logs/ and results/ and writes:
    report/training_curves.csv   one row per epoch per run
    report/test_results.csv      one row per benchmark
    report/summary.md            the tables, ready to paste

Nothing is recomputed and nothing is deleted.

    python compile_results.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "report"

EPOCH = re.compile(
    r"Epoch (\d+) \| Loss: ([\d.]+).*?i2tR1: ([\d.]+).*?i2tR5: ([\d.]+)"
    r".*?i2tR10: ([\d.]+).*?t2iR1: ([\d.]+).*?gamma: ([-\d.]+)"
    r".*?i2t_hnm: ([-\d.]+).*?(?:grad_norm: ([\d.e+-]+) \| )?Time: ([\d.]+)s")
PARAMS = re.compile(r"Image=([\d,]+) \| Text=([\d,]+)")
RECOVER = re.compile(r"Recovered from Epoch: (\d+)")
METRIC = re.compile(r"(coco5k_\w+|att_hard_neg_acc|rel_hard_neg_acc)\s*\|\s*([\d.]+)")

# How each run should be described in the write-up. Anything not listed keeps
# its raw tag.
LABELS = {
    # Qubits per atomic grammatical type, and ansatz layers.
    "fast": "1 qubit/type, 2 layers",
    "long": "1 qubit/type, 2 layers (reference executor)",
    "fastn2": "2 qubits/type, 2 layers",
    "n2": "2 qubits/type, 2 layers (reference executor)",
    "n3r": "3 qubits/type, 2 layers", "n3ring": "3 qubits/type, 2 layers",
    "n4r": "4 qubits/type, 2 layers", "n4ring": "4 qubits/type, 2 layers",
    "n2l3": "2 qubits/type, 3 layers", "n2l4": "2 qubits/type, 4 layers",
    "n2l5": "2 qubits/type, 5 layers", "n2l6": "2 qubits/type, 6 layers",
    "n2l8": "2 qubits/type, 8 layers",
    "n3l3": "3 qubits/type, 3 layers", "n3l4": "3 qubits/type, 4 layers",
    # Two CNOT rings per layer: more entangling reach, no extra parameters.
    "n3r2": "3 qubits/type, 2 layers, repeated entangler",
    # Several captions of the same image treated as positives in one batch.
    "k2": "1 qubit/type, 2 layers, 2 captions per image",
    "k5": "1 qubit/type, 2 layers, 5 captions per image",
    # Width of the shared space where the two encoders meet.
    "n2l4q10": "2 qubits/type, 4 layers, 10 output qubits",
    "n2l4q11": "2 qubits/type, 4 layers, 11 output qubits",
    # Grammar replaced by a commutative product, so word order is invisible.
    "spider": "bag of words (grammar removed)",
    "vqc20k_spider": "bag of words, 20k images",
    "vqc10k_spider": "bag of words, 10k images",
    "tn10k_spider": "classical bag of words, 10k images",
    # Classical tensor-network tower, for comparison.
    "tn5k": "classical, 5k images", "tn10k": "classical, 10k images",
    "tn20k": "classical, 20k images", "tnfull": "classical, full MSCOCO",
    # Loss-term ablations, all at 10k images.
    "nowarp": "loss without the arcsin warp",
    "nopurity": "loss without the purity penalty",
    "nosmooth": "loss without label smoothing",
    "clean": "loss with all three terms removed",
    "temp03": "temperature 0.03", "temp05": "temperature 0.05",
    "temp10": "temperature 0.10",
    "bs512": "batch size 512", "bs768": "batch size 768",
    "l1": "1 ansatz layer", "l3": "3 ansatz layers",
    # Image encoder.
    "amp": "frozen image encoder",
    "v2": "image projection, 54 angles", "v4": "image projection, 108 angles",
    "v8": "image projection, 216 angles", "v16": "image projection, 432 angles",
    # Vocabulary sharing.
    "gen": "shared angle generator",
    "oov": "per-role fallback for unseen words",
    "drop": "symbol dropout", "gendrop": "generator with symbol dropout",
    "gen_oov": "generator with per-role fallback",
}


def tag_of(name: str) -> str:
    m = re.match(r"q-(?:full|vqc\d+k)[-_](.+?)-\d+\.out$", name)
    return m.group(1) if m else name


def read_training():
    rows = []
    for f in sorted((ROOT / "logs").glob("q-*.out")):
        text = f.read_text(errors="ignore").replace("\r", "\n")
        if "Epoch " not in text:
            continue
        tag = tag_of(f.name)
        if "vqc10k" in f.name or "Train=10000" in text:
            tag = f"{tag}@10k"
        pm = PARAMS.search(text)
        img = int(pm.group(1).replace(",", "")) if pm else None
        txt = int(pm.group(2).replace(",", "")) if pm else None
        for line in text.splitlines():
            m = EPOCH.search(line)
            if not m:
                continue
            e, loss, r1, r5, r10, t1, gamma, hnm, grad, t = m.groups()
            rows.append(dict(
                run=tag, label=LABELS.get(tag, tag),
                text_params=txt, image_params=img,
                epoch=int(e), loss=float(loss),
                val_i2t_r1=float(r1), val_i2t_r5=float(r5),
                val_i2t_r10=float(r10), val_t2i_r1=float(t1),
                gamma=float(gamma), hardest_negative_margin=float(hnm),
                grad_norm=float(grad) if grad else None,
                seconds=float(t)))
    return rows


def read_tests():
    rows = []
    for f in sorted(list((ROOT / "results").glob("*.txt"))
                    + list((ROOT / "logs").glob("q-bench*.out"))
                    + list((ROOT / "logs").glob("q-aro*.out"))):
        text = f.read_text(errors="ignore")
        blocks = re.split(r"^#{5,}\s*", text, flags=re.M)
        for block in blocks:
            ep = RECOVER.search(block)
            vals = dict(METRIC.findall(block))
            if not ep or not vals:
                continue
            name = re.match(r"(\S+)", block)
            tag = name.group(1) if name else f.stem
            tag = re.sub(r"^(mscoco-vqcfull-|mscoco-|bench_|vqcfull_)", "", tag)
            tag = re.sub(r"(_grid|_all|_last|_final|_ep\d+)$", "", tag)
            if tag in ("===", "benchmarking") or tag.startswith("#"):
                continue
            row = dict(run=tag, label=LABELS.get(tag, tag),
                       epoch=int(ep.group(1)), source=f.name)
            for k, v in vals.items():
                row[k.replace("coco5k_", "")] = round(float(v) * 100, 2)
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list({k: None for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  {path}  ({len(rows)} rows)")


def main():
    OUT.mkdir(exist_ok=True)
    train, test = read_training(), read_tests()
    write_csv(OUT / "training_curves.csv", train)
    write_csv(OUT / "test_results.csv", test)

    # Best test result per run, by recall sum.
    # One row per configuration, from its latest checkpoint. Taking the
    # best-scoring evaluation instead would be selecting on the test set, and
    # the final checkpoint has been the stronger of the two wherever both were
    # measured. Every measurement stays in test_results.csv.
    best = {}
    for r in test:
        if "i2tR1" not in r:
            continue
        lab = r.get("label") or r["run"]
        if lab not in best or r["epoch"] > best[lab][1]["epoch"]:
            best[lab] = (r.get("i2tR1", 0), r)

    final = {}
    for r in train:
        if r["run"] not in final or r["epoch"] > final[r["run"]]["epoch"]:
            final[r["run"]] = r

    lines = ["# Results", "",
             "Image-to-text recall on the full 5,000-image MSCOCO test set",
             "(24,909 captions). Chance is 0.02%.", "",
             "| configuration | text parameters | epochs | R@1 | R@5 | R@10 |",
             "|---|---|---|---|---|---|"]
    for _, (total, r) in sorted(best.items(), key=lambda x: -x[1][0]):
        f = final.get(r["run"], {})
        p = f.get("text_params")
        lines.append(f"| {r['label']} | {p:,} | {r['epoch']} | "
                     f"{r.get('i2tR1', 0):.2f} | {r.get('i2tR5', 0):.2f} | "
                     f"{r.get('i2tR10', 0):.2f} |" if p else
                     f"| {r['label']} | - | {r['epoch']} | "
                     f"{r.get('i2tR1', 0):.2f} | {r.get('i2tR5', 0):.2f} | "
                     f"{r.get('i2tR10', 0):.2f} |")

    aro = [r for r in test if "att_hard_neg_acc" in r or "rel_hard_neg_acc" in r]
    if aro:
        lines += ["", "## ARO transfer, zero-shot from MSCOCO", "",
                  "Chance is 0.50.", "",
                  "| configuration | epochs | attribution | relation |",
                  "|---|---|---|---|"]
        for r in aro:
            lines.append(f"| {r['label']} | {r['epoch']} | "
                         f"{r.get('att_hard_neg_acc', 0) / 100:.4f} | "
                         f"{r.get('rel_hard_neg_acc', 0) / 100:.4f} |")

    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"  {OUT / 'summary.md'}")
    print(f"\n  {len({r['run'] for r in train})} runs, "
          f"{len(train)} epochs, {len(test)} benchmarks")


if __name__ == "__main__":
    main()
