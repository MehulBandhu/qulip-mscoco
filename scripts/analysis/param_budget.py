"""Where does the model's capacity actually go?

Every distinct (word, grammatical type) gets its own parameters, so the budget
splits three ways: bare nouns, which are pure vocabulary; higher-order words
like adjectives and verbs, which are the compositional machinery; and repeated
spend on inflected forms of the same stem, where "dog" and "dogs" each get
their own block and neither can help the other.

It also measures drift: how far each parameter moved from its initialisation
during training. A group that barely moved was not where learning happened.

    python -m scripts.param_budget -cfg configs/tn10k.yaml -cp <best.pt>

Reinitialises a second copy of the model under the same seed to recover the
starting values, so nothing extra needs to have been saved during training.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

import pandas as pd
import torch

from modules.data_pipeline.engine import DataEngine
from modules.utils.factory import build_experiment
from modules.utils.general import CheckpointManager, setup_exp

# Crude but adequate: COCO captions are plain English and we only need to know
# whether two symbols are forms of the same stem, not to lemmatise properly.
SUFFIXES = ("ing", "ers", "er", "ies", "ied", "es", "ed", "s")


def stem(word: str) -> str:
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            if suf == "ies":
                return base + "y"
            if base.endswith(base[-1] * 2) and len(base) > 3:
                return base[:-1]      # running -> run
            return base
    return word


def parse_symbol(sym: str) -> tuple[str, int]:
    """Return the word and how many wires its type has. Symbols look like
    'street__n' or 'holds__n@out@n_l0_7'."""
    head = re.split(r"_l\d+_\d+$", sym)[0]
    parts = head.split("__", 1)
    word = parts[0].lower()
    types = parts[1] if len(parts) > 1 else ""
    return word, types.count("@") + 1 if types else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-cfg", "--config", required=True)
    ap.add_argument("-cp", "--checkpoint", required=True)
    args = ap.parse_args()

    config, device, _ = setup_exp(args.config)
    ansatz, image_model, text_model, _ = build_experiment(config, device)
    engine = DataEngine(config, ansatz, device)
    train = engine.compile_text("train")
    val = engine.compile_text("val")
    frame = pd.concat([train, val], ignore_index=True)
    engine.text_init(text_model, frame)

    # setup_exp has already seeded, so a second model built the same way starts
    # from the same numbers the trained one did.
    _, _, fresh, _ = build_experiment(config, device)
    engine.text_init(fresh, frame)

    def symmap(model):
        # The classical tower keys a ParameterDict directly; the quantum one
        # keeps a name-to-index map into a flat tensor.
        return getattr(model, "sym2param", None) or model.params

    def tensor_for(model, sym):
        """sym2param holds an index into model.params in both towers, though
        the quantum one indexes a flat tensor and the classical one a list."""
        entry = symmap(model)[sym]
        return model.params[entry] if isinstance(entry, int) else entry

    before = {sym: tensor_for(fresh, sym).detach().clone().reshape(-1)
              for sym in symmap(fresh)}
    quantum = config["model_type"] == "vqc"

    CheckpointManager.load_model(args.checkpoint, image_model, text_model, device)

    # ---- how many parameters each symbol owns, and how far it moved --------
    size, drift = {}, {}
    for sym in symmap(text_model):
        end = tensor_for(text_model, sym).detach().reshape(-1)
        start = before.get(sym)
        size[sym] = end.numel()
        # Relative movement, so a big tensor and a single angle are comparable.
        drift[sym] = ((end - start).norm() / start.norm().clamp(min=1e-9)).item() \
            if start is not None and start.numel() == end.numel() else float("nan")

    # ---- how often each word is actually used -----------------------------
    counts = Counter()
    for col in ("captions_symbols", "captions_diagram"):
        if col in frame.columns:
            for cell in frame[col]:
                items = cell if isinstance(cell, list) else [cell]
                for item in items:
                    if isinstance(item, list):
                        for g in item:
                            if isinstance(g, dict) and "name" in g:
                                counts[g["name"]] += 1
            break

    total = sum(size.values())
    print(f"\n{len(size):,} symbols, {total:,} parameters\n")

    # ---- vocabulary against composition -----------------------------------
    print("by grammatical order")
    print(f"  {'order':>6} {'symbols':>9} {'parameters':>14} {'share':>7} {'mean drift':>11}")
    by_order = defaultdict(lambda: [0, 0, []])
    for sym, n in size.items():
        _, order = parse_symbol(sym)
        by_order[order][0] += 1
        by_order[order][1] += n
        by_order[order][2].append(drift[sym])
    for order in sorted(by_order):
        c, n, d = by_order[order]
        kind = "nouns" if order == 1 else "modifiers" if order == 2 else "verbs etc"
        mean_d = sum(x for x in d if x == x) / max(1, len(d))
        print(f"  {order:>6} {c:>9,} {n:>14,} {n / total:>6.1%} {mean_d:>11.4f}   {kind}")

    lexical = by_order[1][1]
    print(f"\n  vocabulary (order 1): {lexical / total:.1%} of parameters")
    print(f"  composition (order 2+): {1 - lexical / total:.1%}")

    # ---- spend on inflected forms of one stem -----------------------------
    families = defaultdict(list)
    for sym in size:
        word, _ = parse_symbol(sym)
        families[stem(word)].append(sym)

    multi = {k: v for k, v in families.items()
             if len({parse_symbol(s)[0] for s in v}) > 1}
    wasted = sum(size[s] for v in multi.values() for s in v[1:])
    print(f"\ninflection")
    print(f"  {len(multi):,} stems appear in more than one form")
    print(f"  {wasted:,} parameters sit in the duplicates ({wasted / total:.1%})")
    for k in sorted(multi, key=lambda k: -sum(size[s] for s in multi[k]))[:8]:
        forms = sorted({parse_symbol(s)[0] for s in multi[k]})
        print(f"    {k:<14} {', '.join(forms[:6])}")

    # ---- did rare words learn anything? -----------------------------------
    if counts:
        print(f"\nby how often the word appears")
        print(f"  {'uses':>14} {'symbols':>9} {'parameters':>14} {'mean drift':>11}")
        buckets = [(1, 1), (2, 5), (6, 20), (21, 100), (101, 10**9)]
        for lo, hi in buckets:
            syms = [s for s in size if lo <= counts.get(s, 0) <= hi]
            if not syms:
                continue
            n = sum(size[s] for s in syms)
            d = [drift[s] for s in syms if drift[s] == drift[s]]
            band = f"{lo}" if lo == hi else f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            print(f"  {band:>14} {len(syms):>9,} {n:>14,} "
                  f"{sum(d) / max(1, len(d)):>11.4f}")

    print("\nA large share in order 1 with little drift elsewhere means the model")
    print("is spending itself on vocabulary rather than on composition.")


if __name__ == "__main__":
    main()
