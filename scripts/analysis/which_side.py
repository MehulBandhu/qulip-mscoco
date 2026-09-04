"""Substitute CLIP embeddings for one tower at a time, to locate the retrieval gap.

Four scorings of the same 5,000-image test set, swapping one side at a time for
CLIP's own representation:

    quantum text + quantum image   the model as trained
    CLIP text    + quantum image   how good the image side could be
    quantum text + CLIP image      how good the text side could be
    CLIP text    + CLIP image      the ceiling

Whichever substitution recovers more is the side with slack; whichever barely
moves is already saturated. Nothing is trained, this only runs existing
checkpoints and CLIP over the test split.

One wrinkle worth stating: the quantum states are 512 complex amplitudes and
CLIP's are 512 reals, so a mixed pair compares vectors from spaces that were
never aligned to each other. The mixed numbers are therefore a floor on what
each side could do, not a fair estimate, they say "at least this much", and a
large gap between the two mixed rows is the signal.

    python which_side.py -cfg configs/vqcfull_fast.yaml -cp <checkpoint>
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


def recall(sim, mask, ks=(1, 5, 10)):
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

    q_txt, q_img, per_image = [], [], []
    with torch.no_grad():
        for batch in loader:
            img_batch, txt_batch = engine.eval_mapper(batch)
            q_img.append(image_model(img_batch.to(device)).flatten(1).cpu())
            payload = []
            for item in txt_batch:
                if isinstance(item[0], str) or isinstance(item[0][0], dict):
                    item = [item]
                per_image.append(len(item))
                for c in item:
                    payload.append((c[0], c[1]))
            q_txt.append(text_model(payload).flatten(1).cpu())
    q_txt, q_img = torch.cat(q_txt), torch.cat(q_img)
    print(f"{q_img.shape[0]} images, {q_txt.shape[0]} captions")

    diagram_col = next(c for c in compiled.columns if c.endswith("_diagram"))
    captions = []
    for text_row, diag_row in zip(compiled["captions"], compiled[diagram_col]):
        text_row = list(text_row) if isinstance(
            text_row, (list, tuple, np.ndarray)) else [text_row]
        n = len(diag_row) if isinstance(diag_row, (list, tuple)) else 1
        captions.extend(text_row[:n])
    if len(captions) != q_txt.shape[0]:
        raise SystemExit(f"{len(captions)} captions against {q_txt.shape[0]} states")

    print("encoding captions with CLIP")
    import clip
    clip_model, _ = clip.load("ViT-B/32", device="cpu")
    clip_model.eval()
    chunks = []
    with torch.no_grad():
        for s in range(0, len(captions), 256):
            chunks.append(clip_model.encode_text(
                clip.tokenize(captions[s:s + 256], truncate=True)).float())
    c_txt = torch.cat(chunks)

    bank = torch.load(config["dataset"]["train"]["img_path"], map_location="cpu")
    c_img = torch.stack([bank[int(i)].flatten().float()
                         for i in compiled["image_id"]])

    mask = torch.zeros((q_img.shape[0], q_txt.shape[0]), dtype=torch.bool)
    col = 0
    for row, n in enumerate(per_image):
        mask[row, col:col + n] = True
        col += n

    def score(img, txt):
        cx = img.is_complex() or txt.is_complex()
        a = F.normalize(img.to(torch.complex64) if cx else img, dim=-1)
        b = F.normalize(txt.to(torch.complex64) if cx else txt, dim=-1)
        return (a @ b.conj().T).abs().float() if cx else a @ b.T

    rows = [
        ("quantum text + quantum image", q_img, q_txt),
        ("CLIP text    + quantum image", q_img, c_txt),
        ("quantum text + CLIP image   ", c_img, q_txt),
        ("CLIP text    + CLIP image   ", c_img, c_txt),
    ]
    print(f"\n  {'pairing':<30} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    got = {}
    for name, img, txt in rows:
        r = recall(score(img, txt), mask)
        got[name.strip()] = r
        print(f"  {name:<30} {r[0]:>7.2f} {r[1]:>7.2f} {r[2]:>7.2f}")

    base = got["quantum text + quantum image"][0]
    swap_txt = got["CLIP text    + quantum image"][0]
    swap_img = got["quantum text + CLIP image"][0]
    print(f"\n  swapping the text side gains  {swap_txt - base:+.2f} R@1")
    print(f"  swapping the image side gains {swap_img - base:+.2f} R@1")
    if swap_txt > swap_img:
        print("\n  The text tower is the weaker side: replacing it recovers more.")
    elif swap_img > swap_txt:
        print("\n  The image tower is the weaker side: replacing it recovers more.")
    else:
        print("\n  Neither substitution helps, so the two are matched or the")
        print("  representations are too misaligned for this test to say.")


if __name__ == "__main__":
    main()
