"""Train with several captions of the same image as positives in one batch.

Training currently samples one caption per image per epoch. Over 100 epochs
every caption is seen about twenty times, so nothing is wasted, but the five
captions of an image are never present in the same batch, so the model is never
told that different wordings describe the same picture.

This keeps 256 DISTINCT images per batch and takes K captions of each, giving a
[B*K, B] similarity matrix. Packing K captions into a fixed 256-row batch
instead would cut the negative pool from 255 images to 255/K, and the batch-size
experiments already showed fewer negatives hurts.

Two details matter:

  Image-to-text uses soft targets, 1/K on each of that image's captions. A
    logsumexp over the positives would let the model satisfy the loss by ranking
    one caption highly and ignoring the rest.
  The purity penalty must skip same-image caption pairs. It exists to stop
    captions collapsing together, and unmasked it would push apart exactly the
    pairs this objective is trying to bring together.

Point a config at these by path, as the existing config already does for
CocoDataset:

    dataset:
      class: "modules.data_pipeline.multipositive.MultiPositiveDataset2"
      collate_fn: "modules.data_pipeline.multipositive.multi_collate_fn"
      eval_mapper: "modules.data_pipeline.multipositive.multi_mapper"
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.data_pipeline.datasets import CocoDataset


class MultiPositiveDataset(CocoDataset):
    """K captions per image during training, one image per row as before."""

    captions_per_image = 2

    def __getitem__(self, idx):
        row = self.compiled_df.iloc[idx]
        image = self._load_image(row.get('image_id', row.get('file_name')))
        einsums, symbols = row['captions_einsum'], row['captions_symbols']

        if self.mode != "train":
            return {"image": image, "caption": list(zip(einsums, symbols))}

        # Without replacement, so the K captions in one update are genuinely
        # different wordings. Falls back to sampling with replacement if the
        # image has fewer than K parseable captions.
        k = min(self.captions_per_image, len(einsums))
        picks = random.sample(range(len(einsums)), k)
        while len(picks) < self.captions_per_image:
            picks.append(random.randrange(len(einsums)))

        return {"image": image,
                "caption": [(einsums[i], symbols[i]) for i in picks]}


class MultiPositiveDataset2(MultiPositiveDataset):
    captions_per_image = 2


class MultiPositiveDataset5(MultiPositiveDataset):
    captions_per_image = 5


def multi_collate_fn(batch):
    return {"image": torch.stack([b['image'] for b in batch]),
            "caption": [b['caption'] for b in batch]}


def multi_mapper(batch):
    """Leave the nesting alone.

    A row holds a list of that image's captions in both modes. train_epoch
    flattens for the loss, and global_retrieval does its own flattening for
    evaluation, doing it here as well made it treat every caption as a
    separate image.
    """
    return batch["image"], batch["caption"]


class MultiPositive_InfoNCE(nn.Module):
    """Fubini-Study InfoNCE with K captions per image.

    Same warp, temperature, label smoothing and purity penalty as FS_InfoNCE;
    only the target structure and the purity mask change.
    """

    def __init__(self, temperature: float = 0.07, lambda_reg: float = 0.1,
                 label_smoothing: float = 0.1, eps: float = 1e-7,
                 warp: bool = True):
        super().__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.warp = warp

    def _angular(self, a, b):
        ov = torch.clamp((a @ b.conj().t()).abs(), 0.0, 1.0 - self.eps)
        if self.warp:
            ov = 0.5 + (0.5 * ov)
        return (torch.asin(ov) / (math.pi / 2)) / self.temperature

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor):
        n_txt, n_img = text_emb.size(0), image_emb.size(0)
        if n_txt % n_img != 0:
            raise ValueError(f"{n_txt} captions is not a whole multiple of "
                             f"{n_img} images")
        k = n_txt // n_img
        device = text_emb.device

        text_emb = F.normalize(
            text_emb.flatten(start_dim=1).to(torch.complex64), dim=1, p=2)
        image_emb = F.normalize(
            image_emb.flatten(start_dim=1).to(torch.complex64), dim=1, p=2)

        sim = self._angular(text_emb, image_emb)          # [B*K, B]

        # Caption to image: exactly one right answer each.
        owner = torch.arange(n_img, device=device).repeat_interleave(k)
        loss_t2i = F.cross_entropy(sim, owner,
                                   label_smoothing=self.label_smoothing)

        # Image to caption: K right answers, weighted 1/K so the model has to
        # rank all of them rather than just the easiest one.
        log_p = F.log_softmax(sim.t(), dim=1)             # [B, B*K]
        target = torch.zeros_like(log_p)
        rows = torch.arange(n_img, device=device).repeat_interleave(k)
        target[rows, torch.arange(n_txt, device=device)] = 1.0 / k
        if self.label_smoothing > 0:
            target = (target * (1 - self.label_smoothing)
                      + self.label_smoothing / n_txt)
        loss_i2t = -(target * log_p).sum(dim=1).mean()

        loss = 0.5 * (loss_t2i + loss_i2t)

        if self.lambda_reg > 0 and n_txt > 1:
            overlap = torch.clamp(
                (text_emb @ text_emb.conj().t()).abs(), 0.0, 1.0 - self.eps)
            closeness = 1.0 - (torch.acos(overlap) / (math.pi / 2))
            # Skip the diagonal and any pair of captions describing the same
            # image, separating those is what this objective works against.
            same = owner[:, None] == owner[None, :]
            loss = loss + self.lambda_reg * closeness[~same].mean()

        return loss
