"""Break test-set recall down by caption length, circuit size and vocabulary coverage.

Scores the test set once, keeps the rank of every caption, and joins that to
what the caption looks like, how long it is, how wide its circuit is, whether
it contains words the model never saw, and which grammatical types it uses.

The aggregate recall hides all of this. A model that handles short captions and
collapses on long ones has the same R@1 as one that is uniformly mediocre, and
they call for different fixes.

Writes report/per_caption.csv and report/error_analysis.md.

    python error_analysis.py -cfg configs/vqcfull_n3l4.yaml -cp <checkpoint>
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

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
    ap.add_argument("--split", default="coco5k")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or Path(args.config).stem

    config, device, _ = setup_exp(args.config)
    ansatz, image_model, text_model, _ = build_experiment(config, device)
    engine = DataEngine(config, ansatz, device)
    train = engine.compile_text("train")
    val = engine.compile_text("val")
    engine.text_init(text_model, pd.concat([train, val], ignore_index=True))
    CheckpointManager.load_model(args.checkpoint, image_model, text_model, device)
    text_model.eval(); image_model.eval()

    # Which symbols training ever saw, for the unseen-word split below.
    seen = set(text_model.symbols)

    compiled = engine.compile_text(args.split)
    loader = engine.get_loader(compiled, split=args.split)

    txt, img, per_image = [], [], []
    with torch.no_grad():
        for batch in loader:
            img_batch, txt_batch = engine.eval_mapper(batch)
            img.append(image_model(img_batch.to(device)).flatten(1).cpu())
            payload = []
            for item in txt_batch:
                if isinstance(item[0], str) or isinstance(item[0][0], dict):
                    item = [item]
                per_image.append(len(item))
                for c in item:
                    payload.append((c[0], c[1]))
            txt.append(text_model(payload).flatten(1).cpu())
    txt, img = torch.cat(txt), torch.cat(img)
    print(f"  {img.shape[0]} images, {txt.shape[0]} captions")

    a = F.normalize(img.to(torch.complex64), dim=-1)
    b = F.normalize(txt.to(torch.complex64), dim=-1)
    sim = (a @ b.conj().T).abs().float()          # [images, captions]

    # Rank of each caption for its own image, and rank of the right image for
    # each caption. Both directions, since they fail differently.
    owner, col = [], 0
    for row, n in enumerate(per_image):
        owner.extend([row] * n)
        col += n
    owner = torch.tensor(owner)

    caption_rank = (sim.T > sim.T[torch.arange(len(owner)), owner]
                    .unsqueeze(1)).sum(1)         # rank of the true image
    order = sim.argsort(dim=1, descending=True)

    # For each caption, where does it sit among all captions for its image?
    pos_in_image = torch.empty(len(owner), dtype=torch.long)
    for i in range(sim.shape[0]):
        ranked = order[i]
        place = {int(c): r for r, c in enumerate(ranked.tolist())}
        for c in (owner == i).nonzero().flatten().tolist():
            pos_in_image[c] = place[c]

    # Caption-level features. The circuit is one operand per gate or word
    # depending on the executor, so record both length and operand count.
    rows, k = [], 0
    diagram_col = next(c for c in compiled.columns if c.endswith("_diagram"))
    symbol_col = next(c for c in compiled.columns if c.endswith("_symbols"))
    for _, r in compiled.iterrows():
        texts = list(r["captions"]) if hasattr(r["captions"], "__len__") else [r["captions"]]
        diagrams = r[diagram_col] if isinstance(r[diagram_col], list) else [r[diagram_col]]
        symbols = r[symbol_col] if isinstance(r[symbol_col], list) else [r[symbol_col]]
        for j in range(len(diagrams)):
            if k >= len(owner):
                break
            names = []
            def walk(v):
                if isinstance(v, dict):
                    if v.get("name"):
                        names.append(v["name"])
                    for x in v.values():
                        walk(x)
                elif isinstance(v, list):
                    for x in v:
                        walk(x)
            walk(symbols[j])
            unseen = sum(1 for n in names if n not in seen)
            types = Counter(n.split("__", 1)[1].split("_l")[0]
                            for n in names if "__" in n)
            rows.append(dict(
                caption=texts[j] if j < len(texts) else "",
                words=len(str(texts[j]).split()) if j < len(texts) else 0,
                operands=len(names),
                distinct_symbols=len(set(names)),
                unseen_symbols=unseen,
                max_type_arity=max((t.count("@") + 1 for t in types), default=1),
                rank_of_image=int(caption_rank[k]),
                rank_among_captions=int(pos_in_image[k]),
            ))
            k += 1

    REPORT.mkdir(exist_ok=True)
    out = REPORT / f"per_caption_{tag}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"  {out}  ({len(rows)} captions)")

    df = pd.DataFrame(rows)
    df["hit1"] = df["rank_among_captions"] < 1
    df["hit10"] = df["rank_among_captions"] < 10

    md = [f"# Error analysis: {tag}", "",
          f"{len(df):,} captions over {img.shape[0]:,} images.", ""]

    def table(col, bins, title, fmt="{}"):
        md.append(f"## {title}")
        md.append("")
        md.append("| range | captions | top 1 (%) | top 10 (%) |")
        md.append("|---|---|---|---|")
        cut = pd.cut(df[col], bins=bins, include_lowest=True)
        for interval, g in df.groupby(cut, observed=True):
            md.append(f"| {fmt.format(interval)} | {len(g):,} | "
                      f"{100 * g['hit1'].mean():.2f} | "
                      f"{100 * g['hit10'].mean():.2f} |")
        md.append("")

    table("words", [0, 6, 8, 10, 12, 15, 100], "By caption length (words)")
    table("operands", [0, 100, 200, 300, 400, 100000],
          "By circuit size (parameterised operands)")
    table("max_type_arity", [0, 1, 2, 3, 4, 100],
          "By the widest grammatical type in the caption")

    md += ["## Captions containing words the model never saw", "",
           "| | captions | top 1 (%) | top 10 (%) |", "|---|---|---|---|"]
    for name, g in (("all symbols seen", df[df.unseen_symbols == 0]),
                    ("one or more unseen", df[df.unseen_symbols > 0])):
        if len(g):
            md.append(f"| {name} | {len(g):,} | {100 * g['hit1'].mean():.2f} | "
                      f"{100 * g['hit10'].mean():.2f} |")
    md.append("")

    # Correlations, on ranks so a few catastrophic failures do not dominate.
    md += ["## Correlation with the rank of the true caption", "",
           "Spearman, so a handful of very bad cases cannot dominate.", "",
           "| feature | correlation |", "|---|---|"]
    for col in ("words", "operands", "distinct_symbols", "unseen_symbols",
                "max_type_arity"):
        rho = df[col].corr(df["rank_among_captions"], method="spearman")
        md.append(f"| {col.replace('_', ' ')} | {rho:+.3f} |")
    md += ["", "A positive value means the caption is ranked worse as that "
               "feature grows.", ""]

    (REPORT / f"error_analysis_{tag}.md").write_text("\n".join(md) + "\n")
    print(f"  report/error_analysis_{tag}.md")


if __name__ == "__main__":
    main()
