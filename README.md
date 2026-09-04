# CCG-VQC on MSCOCO

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
