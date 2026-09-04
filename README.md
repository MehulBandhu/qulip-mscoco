# Quantum text encoders on MSCOCO

A compositional quantum circuit that reads image captions, trained and evaluated
on the full MSCOCO retrieval benchmark. Each caption is parsed to a CCG tree,
every word becomes a small parameterised quantum circuit whose width is set by
its grammatical type, composition follows the parse, and the resulting state is
matched against a frozen CLIP image embedding. Everything is simulated
classically as tensor contraction.

**7.36% of captions ranked first out of 24,909, with 4.8 million parameters.**
The classical tensor-network model on the same data and the same split reaches
1.72% with 2.3 billion.

<p align="center">
  <img src="report/figures/F1_model_size.png" width="46%">
  <img src="report/figures/F2_depth_and_width.png" width="53%">
</p>

Left: accuracy against model size, with the classical baseline marked. The fit
is over the width series only, since depth and width behave differently and one
exponent through both would hide it. Right: the same nine runs as a grid, and
against parameter count.

## Results

Image-to-text recall on all 5,000 test images and 24,909 captions. Chance is
0.02%.

| configuration | parameters | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| 2 qubits/type, 5 layers | 4,797,556 | **7.36** | **22.84** | **33.84** |
| 3 qubits/type, 4 layers | 5,363,041 | 6.46 | 21.54 | 33.26 |
| 2 qubits/type, 4 layers | 3,838,045 | 6.22 | 21.62 | 32.88 |
| 4 qubits/type, 2 layers | 3,444,019 | 5.38 | 17.54 | 26.90 |
| 2 qubits/type, 3 layers | 2,878,534 | 4.98 | 19.48 | 30.26 |
| 1 qubit/type, 2 layers | 1,156,525 | 3.08 | 12.46 | 20.10 |
| classical tensor network | 2,310,327,176 | 1.72 | 6.30 | 9.84 |

Every quantum row is the final checkpoint of a 100-epoch run, evaluated once.
The classical tower ran 30 epochs; its validation recall had been flat for its
last four (0.065, 0.067, 0.068, 0.065) while its loss was still falling, so it
had stopped improving at retrieval before it stopped training. Full
tables are in [`report/summary.md`](report/summary.md); every measurement,
including intermediate checkpoints, is in
[`report/test_results.csv`](report/test_results.csv).

## What the experiments found

**Capacity is the lever that works.** Accuracy rises with parameter count across
nine configurations. Depth and width are not interchangeable: at two qubits per
type, adding layers is the better use of a parameter budget, and `2 qubits, 5
layers` beats `3 qubits, 4 layers` outright while being 10% smaller. The
confidence intervals overlap, so this is a comparison between specific
configurations rather than a fitted law.

**Trainability is not the limit.** Resampling the text tower 10,000 times
without training, the spread of the loss is flat as circuits widen: a barren
plateau over this range predicts variance falling 304-fold, and the measurement
moves by a factor of 1.05.

<p align="center">
  <img src="report/figures/F5_loss_spread.png" width="42%">
  <img src="report/figures/F4_grammar_ablation.png" width="57%">
</p>

**The grammar is not what buys retrieval.** Replacing the parse with a
commutative product, so word order becomes invisible, matches the best grammar
model on a sixth of the data. Where the grammar does show up is word order
itself: on ARO the flattened variant sits 23 to 25 standard errors *below*
chance, meaning it systematically prefers the corrupted caption, while the
grammar model is above chance on relations.

**Nothing else moved the number.** Fourteen changes to the loss, the image
encoder, the vocabulary sharing and the optimiser were tested; all scored below
the unmodified baseline. The three apparent deviations from the published loss
turn out to be load-bearing, the purity term most of all.

## Making it run

The original executor passes every gate to `opt_einsum` separately, about 284
operands per caption. Two rewrites, each checked against the one before it:

- **word-first** (`src/executors/compact_exec.py`) contracts each word's gates
  into a single tensor before the sentence network, cutting operands to about
  ten and epoch time from 185 minutes to 13
- **tensor ring** (`src/executors/ring_sentence.py`) represents a word as cores
  of bond dimension `2^L` rather than a `2^N` statevector. A CNOT ring is a
  running XOR, so this is exact. At 21 wires it is 672 numbers instead of
  2,097,152, and it is what makes the wider circuits affordable at all

Forward values agree to 1e-7, gradients to 1e-5, and training losses match to
four decimal places over 47 matched epochs. The checks are in
[`scripts/verify/`](scripts/verify).

## Fixes to the upstream code

Seven, of which four blocked training outright: the contrastive loss did not
normalise its embeddings, so image magnitude acted as a per-sample temperature;
the optimiser walked a million separate tensors per step; contraction paths were
searched with the batch axis attached; and the benchmark registered test
vocabulary before loading weights, silently scoring a randomly initialised
model. Collected in
[`scripts/patches/apply_fixes.py`](scripts/patches/apply_fixes.py).

## Reproducing

Both run from the repository root.

```bash
python scripts/report/compile_results.py   # logs and results -> report/*.csv
python scripts/report/figures.py           # csv -> report/figures/*.pdf
```

Training and evaluation run through `scripts/jobs/`, one Slurm script per
configuration, with the matching YAML in `configs/`.

## Layout

```
src/executors/     the three ways of evaluating the circuits
src/training/      multi-positive dataset and loss
scripts/patches/   fixes and features applied to the fork
scripts/analysis/  diagnostics that do not train anything
scripts/verify/    equivalence checks between executors
scripts/report/    result harvesting and figures
scripts/jobs/      Slurm submission scripts
configs/           one YAML per experiment
report/            compiled tables and figures
```

## Caveats

Every configuration is a single run, so differences below about 0.3 in R@1 are
inside the variance you would expect. Checkpoints were selected on a
1,000-image subset of the test set; the reported figures use the final
checkpoint, which does not depend on that selection. Configurations were
compared on the full test set during development. Published MSCOCO numbers use
the Karpathy split while these use `val2017`, which zero-shot CLIP scores 48.4
against a published 50.0.

Built on [QuLIP](https://github.com/quantum-learning-labs/QuLIP).
