"""Rewrite module docstrings and fix stale paths.

Run from the repository root. Reports what it changed and what it could not
find.

    python tidy_docs.py --check
    python tidy_docs.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# First line of each module, as a statement of what the file does.
OPENERS = {
    "scripts/analysis/error_analysis.py":
        "Break test-set recall down by caption length, circuit size and vocabulary coverage.",
    "scripts/analysis/loss_concentration.py":
        "Measure the spread of the loss over random initialisations, as a function of circuit width.",
    "scripts/analysis/which_side.py":
        "Substitute CLIP embeddings for one tower at a time, to locate the retrieval gap.",
    "scripts/analysis/param_budget.py":
        "Report how the parameter count divides across grammatical orders and word frequencies.",
    "scripts/analysis/permutation_sim.py":
        "Compare a caption's state with that of its word-order-corrupted variant.",
    "scripts/analysis/fuse_clip.py":
        "Sweep a linear mixture of the quantum and CLIP similarity matrices.",
    "scripts/analysis/profile_cost.py":
        "Time each stage of a training batch: mapping, both towers, loss and optimiser.",
    "scripts/report/figures.py":
        "Produce the figures in report/figures from the compiled CSVs.",
    "src/executors/compact_exec.py":
        "Contract each word's gates into one tensor before the sentence network.",
    "src/executors/tensor_ring.py":
        "Represent a word circuit as tensor-ring cores of bond dimension 2^L.",
    "src/executors/fast_word.py":
        "Apply a CNOT ring as a single permutation of the statevector.",
    "src/executors/ring_sentence.py":
        "Feed tensor-ring cores into the sentence contraction.",
    "src/training/multipositive.py":
        "Train with several captions of the same image as positives in one batch.",
}

# make_repo.py sorted a flat script directory into subfolders and the docstrings
# still name the old module paths.
MODULE_PATHS = {
    "scripts.apply_fixes": "scripts/patches/apply_fixes.py",
    "scripts.integrate_compact": "scripts/patches/integrate_compact.py",
    "scripts.integrate_ring": "scripts/patches/integrate_ring.py",
    "scripts.add_repeats": "scripts/patches/add_repeats.py",
    "scripts.add_generator": "scripts/patches/add_generator.py",
    "scripts.add_lemma": "scripts/patches/add_lemma.py",
    "scripts.add_symbol_dropout": "scripts/patches/add_symbol_dropout.py",
    "scripts.add_positional": "scripts/patches/add_positional.py",
    "scripts.loss_surface": "scripts/analysis/loss_surface.py",
    "scripts.param_budget": "scripts/analysis/param_budget.py",
    "scripts.profile_cost": "scripts/analysis/profile_cost.py",
    "scripts.surface3d": "scripts/analysis/surface3d.py",
}

# Statements that describe a version history rather than the file.
STALE = {
    "src/executors/compact_exec.py": [
        ("Nothing here is wired into the model yet.",
         "Wired in behind the text.compact config flag by "
         "scripts/patches/integrate_compact.py."),
    ],
}


def replace_opener(text: str, line: str) -> str:
    """Swap the first line of the module docstring."""
    m = re.match(r'^("""|\'\'\')(.*?)\n', text)
    if not m:
        return text
    return f'{m.group(1)}{line}\n' + text[m.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = Path(".")
    changed, missing = [], []

    for rel, opener in OPENERS.items():
        p = root / rel
        if not p.exists():
            missing.append(rel)
            continue
        s = original = p.read_text()
        s = replace_opener(s, opener)

        for old, new in STALE.get(rel, []):
            s = s.replace(old, new)

        if s != original and not args.check:
            p.write_text(s)
        if s != original:
            changed.append(rel)

    # Stale module paths, and hyphens standing in for an em dash.
    for p in list(root.glob("scripts/**/*.py")) + list(root.glob("src/**/*.py")):
        s = original = p.read_text()
        for mod, path in MODULE_PATHS.items():
            s = s.replace(f"python -m {mod}", f"python {path}")
        # " - " inside a comment or docstring reads as an em dash; commas or
        # parentheses say the same thing without the tell.
        s = re.sub(r"^(\s*#.*?) - ", r"\1, ", s, flags=re.M)
        if s != original:
            if not args.check:
                p.write_text(s)
            if str(p) not in changed:
                changed.append(str(p))

    # An identity permutation left over from a version where the axes moved.
    p = root / "src/executors/tensor_ring.py"
    if p.exists():
        s = p.read_text()
        dead = "        cores = cores.permute(0, 1, 2, 3, 4, 5, 6).reshape(\n"
        if dead in s:
            s = s.replace(dead, "        cores = cores.reshape(\n")
            if not args.check:
                p.write_text(s)
            changed.append("src/executors/tensor_ring.py (identity permute)")

    print(f"  {len(changed)} files {'would change' if args.check else 'changed'}")
    for c in sorted(set(changed)):
        print(f"    {c}")
    if missing:
        print(f"\n  {len(missing)} not found:")
        for m in missing:
            print(f"    {m}")


if __name__ == "__main__":
    main()
