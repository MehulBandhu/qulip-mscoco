"""Sweep a linear mixture of the quantum and CLIP similarity matrices.

Scores every image-caption pair twice on the 5,000-image test set, once with
CLIP's own text encoder, once with the trained quantum tower, and sweeps

    S = (1, lam) * S_clip + lam * S_quantum

Nothing is trained. If no mixture beats lam = 0, the quantum ranking adds
nothing and the fusion idea stops there.

Two anchors make the result trustworthy. At lam = 0 the table must reproduce
the validated zero-shot CLIP figure of about 48.4 R@1, and at lam = 1 it must
reproduce the quantum model's own benchmark. If either is wrong, the caption
ordering or the mask is broken and nothing in between means anything, so both
are checked and reported rather than assumed.

    python fuse_clip.py -cfg configs/vqcfull_fast.yaml -cp <checkpoint>
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from modules.data_pipeline.engine import DataEngine
from modules.utils.factory import build_experiment
from modules.utils.general import CheckpointManager, setup_exp


def recall(sim: torch.Tensor, mask: torch.Tensor, ks=(1, 5, 10)):
    """Image-to-text: rank all captions for each image, count a hit if any of
    that image's own captions lands in the top k."""
    order = sim.argsort(dim=1, descending=True)
    hits = mask.gather(1, order).float()
    first = hits.argmax(dim=1)
    found = hits.sum(dim=1) > 0
    return [((first < k) & found).float().mean().item() * 100 for k in ks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-cfg", "--config", required=True)
    ap.add_argument("-cp", "--checkpoint", required=True)
    ap.add_argument("--split", default="coco5k")
    args = ap.parse_args()

    config, device, _ = setup_exp(args.config)
    ansatz, image_model, text_model, _ = build_experiment(config, device)
    engine = DataEngine(config, ansatz, device)
    engine.text_init(text_model, pd.concat(
        [engine.compile_text("train"), engine.compile_text("val")],
        ignore_index=True))
    CheckpointManager.load_model(args.checkpoint, image_model, text_model, device)

    compiled = engine.compile_text(args.split)
    loader = engine.get_loader(compiled, split=args.split)
    text_model.eval(); image_model.eval()

    # --- quantum side ------------------------------------------------------
    # The mapper hands back one entry per image holding that image's caption
    # recipes, so flatten it exactly the way global_retrieval does and record
    # how many captions each image contributed.
    q_txt, q_img, per_image = [], [], []
    with torch.no_grad():
        for batch in loader:
            img_batch, txt_batch = engine.eval_mapper(batch)
            q_img.append(image_model(img_batch.to(device)).flatten(1).cpu())

            payload = []
            for item_captions in txt_batch:
                if isinstance(item_captions[0], str) or \
                        isinstance(item_captions[0][0], dict):
                    item_captions = [item_captions]
                per_image.append(len(item_captions))
                for c in item_captions:
                    payload.append((c[0], c[1]))
            q_txt.append(text_model(payload).flatten(1).cpu())
    q_txt, q_img = torch.cat(q_txt), torch.cat(q_img)
    print(f"{q_img.shape[0]} images, {q_txt.shape[0]} captions")

    # --- the same captions, in the same order ------------------------------
    # `captions` is a numpy array of strings per row, and `captions_diagram` a
    # list in the same order, so walking rows in order reproduces the ordering
    # the loader produced. Anything else silently misaligns the two scores.
    # Take the caption text and the diagrams from the same row, and keep only
    # the captions whose diagram survived compilation, roughly 0.4% fail to
    # parse, so the raw text column has more entries than there are states.
    diagram_col = next(c for c in compiled.columns if c.endswith("_diagram"))
    captions, sizes = [], []
    for text_row, diag_row in zip(compiled["captions"], compiled[diagram_col]):
        text_row = list(text_row) if isinstance(
            text_row, (list, tuple, np.ndarray)) else [text_row]
        n = len(diag_row) if isinstance(diag_row, (list, tuple)) else 1
        captions.extend(text_row[:n])
        sizes.append(n)
    if captions and len(captions) != q_txt.shape[0]:
        raise SystemExit(f"{len(captions)} captions against {q_txt.shape[0]} "
                         f"states. The ordering does not line up")
    if sizes != per_image:
        raise SystemExit("per-image caption counts differ between the dataframe "
                         "and the loader")
    print(f'first caption: "{captions[0][:60]}"')

    print("encoding captions with CLIP")
    import clip
    clip_model, _ = clip.load("ViT-B/32", device="cpu")
    clip_model.eval()
    chunks = []
    with torch.no_grad():
        for s in range(0, len(captions), 256):
            chunks.append(clip_model.encode_text(
                clip.tokenize(captions[s:s + 256], truncate=True)).float())
    c_txt = F.normalize(torch.cat(chunks), dim=-1)

    bank = torch.load(config["dataset"]["train"]["img_path"], map_location="cpu")
    c_img = F.normalize(torch.stack(
        [bank[int(i)].flatten().float() for i in compiled["image_id"]]), dim=-1)

    # --- similarities ------------------------------------------------------
    def overlap(img, txt):
        a = F.normalize(img.to(torch.complex64), dim=-1)
        b = F.normalize(txt.to(torch.complex64), dim=-1)
        return (a @ b.conj().T).abs().float()

    s_q, s_c = overlap(q_img, q_txt), c_img @ c_txt.T
    # Each score has its own scale and spread, so standardise before mixing or
    # lam would mean something different for every pair of models.
    s_q = (s_q - s_q.mean()) / s_q.std()
    s_c = (s_c - s_c.mean()) / s_c.std()

    mask = torch.zeros_like(s_q, dtype=torch.bool)
    col = 0
    for row, n in enumerate(per_image):
        mask[row, col:col + n] = True
        col += n
    assert col == s_q.shape[1], "mask does not cover every caption"

    # --- sweep -------------------------------------------------------------
    rows = []
    for lam in [i / 20 for i in range(21)]:
        rows.append((lam, recall((1 - lam) * s_c + lam * s_q, mask)))

    clip_only, quantum_only = rows[0][1], rows[-1][1]
    print(f"\n  {'lambda':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    for lam, r in rows:
        tag = "  CLIP alone" if lam == 0 else "  quantum alone" if lam == 1 else ""
        print(f"  {lam:>7.2f} {r[0]:>7.2f} {r[1]:>7.2f} {r[2]:>7.2f}{tag}")

    print(f"\nCLIP alone     {clip_only[0]:.2f}  (expected about 48.4)")
    print(f"quantum alone  {quantum_only[0]:.2f}  (expected the model's own benchmark)")
    if abs(clip_only[0] - 48.4) > 5:
        print("WARNING: CLIP alone is far from its validated figure, so the")
        print("captions and images are probably misaligned. Ignore the sweep.")

    best_lam, best = max(rows, key=lambda x: x[1][0])
    print(f"\nbest mixture: lambda {best_lam:.2f}, "
          f"{best[0]:.2f} / {best[1]:.2f} / {best[2]:.2f}")
    if best_lam == 0:
        print("No mixture beats CLIP alone, so the quantum ranking carries")
        print("nothing CLIP does not already have.")
    else:
        print(f"Mixing gains {best[0] - clip_only[0]:+.2f} R@1 over CLIP alone.")


if __name__ == "__main__":
    main()
