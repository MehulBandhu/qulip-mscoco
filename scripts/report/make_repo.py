"""Assemble a clean repository from the working directory.

Copies rather than moves, so the eight running jobs are untouched. Anything
missing is reported and skipped.

    python make_repo.py --dest ~/qulip-mscoco
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

SRC = Path("/cephfs/mbandhu/qulip")

# Our additions to the QuLIP fork, as opposed to Tilen's original modules.
LAYOUT = {
    "src/executors": [
        "compact_exec.py", "ring_sentence.py", "tensor_ring.py", "fast_word.py",
    ],
    "src/training": [
        "modules/data_pipeline/multipositive.py",
    ],
    "scripts/patches": [
        "scripts/apply_fixes.py", "scripts/integrate_compact.py",
        "scripts/integrate_ring.py", "scripts/add_repeats.py",
        "scripts/add_generator.py", "scripts/add_symbol_dropout.py",
        "scripts/add_lemma.py", "scripts/add_positional.py",
    ],
    "scripts/analysis": [
        "loss_concentration.py", "fuse_clip.py", "which_side.py",
        "permutation_sim.py", "postproc.py", "oov_full.py", "ring_cost.py",
        "qubit_cost.py", "type_merge.py", "prefix_share.py", "arity_check.py",
        "scripts/loss_surface.py", "scripts/param_budget.py",
        "scripts/profile_cost.py", "scripts/surface3d.py",
    ],
    "scripts/verify": [
        "ring_check.py", "compact_check.py", "grad_parity.py", "ring_grad.py",
        "grad_cost.py", "check_oov.py",
    ],
    "scripts/report": [
        "compile_results.py", "paper_figures.py", "figures.py", "figures2.py",
        "compare_runs.py", "scripts/graphs.py",
    ],
    "scripts/jobs": [],          # filled by glob below
    "configs": [],               # ditto
}

README = """# CCG-VQC on MSCOCO

Extends the [QuLIP](https://github.com/quantum-learning-labs/QuLIP) quantum text
encoder from ARO to MSCOCO retrieval, and measures how it scales.

A caption is parsed to a CCG tree, each word becomes a small parameterised
quantum circuit, composition follows the parse, and the resulting 512-amplitude
state is matched against a frozen CLIP image embedding. Everything is simulated
classically as tensor contraction.

## Layout

    src/executors/    faster ways to evaluate the circuits (see below)
    src/training/     multi-positive dataset and loss
    scripts/patches/  fixes and features applied to the QuLIP fork
    scripts/analysis/ diagnostics that do not train anything
    scripts/verify/   equivalence checks between executors
    scripts/report/   result harvesting and figures
    scripts/jobs/     Slurm submission scripts
    configs/          one YAML per experiment
    report/           compiled tables and figures

## Executors

Three ways to evaluate the same model, each verified against the last:

- **gate-by-gate**, the original: every gate is its own operand, ~284 per
  caption
- **compact** (`src/executors/compact_exec.py`): each word's gates contracted
  into one tensor first, ~10 operands per caption, 14x faster at one qubit per
  type. Set `text.compact: true`
- **tensor ring** (`src/executors/ring_sentence.py`): each word as cores of
  bond dimension 2^L rather than a 2^N statevector. Exact, and the only one that
  stays affordable as the circuits widen. Set `text.ring: true`

Equivalence is checked in `scripts/verify/`: forward values to 1e-7, gradients
to 1e-5, and identical training losses over matched epochs.

## Reproducing the tables

    python scripts/report/compile_results.py    # logs -> report/*.csv
    python scripts/report/paper_figures.py      # csv -> report/fig*.pdf

## Fixes to the upstream repo

`scripts/patches/apply_fixes.py` collects the corrections needed before any of
this trains. The consequential ones are listed in `report/fixes.md`.
"""

GITIGNORE = """__pycache__/
*.pyc
.venv/
venv/

# Too large for git: embeddings, parsed circuits, checkpoints, raw logs.
data/
checkpoints/
logs/
runs/
*.pkl
*.pt
*.zip

# Keep the compiled tables and figures, not the intermediates.
report/*.npz
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path.home() / "qulip-mscoco"))
    ap.add_argument("--init-git", action="store_true")
    args = ap.parse_args()
    dest = Path(args.dest)

    copied, missing = 0, []
    for folder, files in LAYOUT.items():
        (dest / folder).mkdir(parents=True, exist_ok=True)
        for rel in files:
            s = SRC / rel
            if not s.exists():
                missing.append(rel)
                continue
            shutil.copy2(s, dest / folder / Path(rel).name)
            copied += 1

    for pattern, folder in (("job_*.sh", "scripts/jobs"),
                            ("bench*.sh", "scripts/jobs"),
                            ("*.sh", "scripts/jobs"),
                            ("configs/*.yaml", "configs")):
        for s in sorted(SRC.glob(pattern)):
            target = dest / folder / s.name
            if not target.exists():
                shutil.copy2(s, target)
                copied += 1

    # Compiled results are small and worth versioning.
    if (SRC / "report").exists():
        shutil.copytree(SRC / "report", dest / "report", dirs_exist_ok=True)

    (dest / "README.md").write_text(README)
    (dest / ".gitignore").write_text(GITIGNORE)

    print(f"  {copied} files copied to {dest}")
    if missing:
        print(f"  {len(missing)} not found:")
        for m in missing[:12]:
            print(f"    {m}")

    if args.init_git:
        subprocess.run(["git", "init", "-q"], cwd=dest)
        subprocess.run(["git", "add", "-A"], cwd=dest)
        print("\n  git initialised and staged. To publish:")
        print("    cd", dest)
        print("    git commit -m 'CCG-VQC on MSCOCO: executors, experiments, results'")
        print("    gh repo create qulip-mscoco --private --source=. --push")


if __name__ == "__main__":
    main()
