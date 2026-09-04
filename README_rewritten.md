# Quantum text encoders on MSCOCO

This repository extends [QCLIP](https://github.com/quantum-learning-labs/QCLIP)
to the full MSCOCO retrieval benchmark, and reports how the compositional
quantum text encoder scales with parameter count, circuit depth, circuit width
and training-set size.

Each caption is parsed to a CCG derivation. Every word becomes a parameterised
quantum circuit whose width is set by its grammatical type, composition follows
the parse, and the resulting state is scored against a frozen CLIP image
embedding. All circuits are simulated classically as tensor contraction; no
quantum hardware is involved.

The largest configuration ranks the correct caption first for 7.36% of test
images out of 24,909 candidate captions, using 4.8 million text-encoder
parameters. The classical tensor-network encoder trained on the same data and
evaluated on the same split reaches 1.72% with 2.31 billion parameters.

<p align="center">
  <img src="report/figures/F1_model_size.png" width="46%">
  <img src="report/figures/F2_depth_and_width.png" width="53%">
</p>

Left: accuracy against text-encoder size. The fitted line covers the width
series only, because depth and width scale differently and a single exponent
through both obscures that. Right: the same nine runs as a grid, and plotted
against parameter count.

## Installation

This repository is an overlay on a QCLIP checkout rather than a standalone
package. It imports `modules.*` from QCLIP in 25 files and patches four QCLIP
source files in place.

```bash
git clone https://github.com/quantum-learning-labs/QCLIP.git qulip
cd qulip
git clone https://github.com/MehulBandhu/qulip-mscoco.git overlay
cp -r overlay/{src,scripts,configs} .
python -m pip install -r overlay/requirements.txt
python scripts/patches/apply_fixes.py
```

`apply_fixes.py` is idempotent and reports which fixes were already present.
All commands below are run from the QCLIP root.

## Results

Image-to-text recall on all 5,000 MSCOCO `val2017` test images and 24,909
captions. Chance is 0.02%. Each configuration is a single run.

| configuration | text parameters | epochs | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|
| 2 qubits/type, 5 layers | 4,797,556 | 100 | 7.36 | 22.84 | 33.84 |
| 3 qubits/type, 4 layers | 5,363,041 | 100 | 6.46 | 21.54 | 33.26 |
| 2 qubits/type, 4 layers | 3,838,045 | 100 | 6.22 | 21.62 | 32.88 |
| 4 qubits/type, 2 layers | 3,444,019 | 100 | 5.38 | 17.54 | 26.90 |
| 2 qubits/type, 3 layers | 2,878,534 | 100 | 4.98 | 19.48 | 30.26 |
| 1 qubit/type, 2 layers | 1,156,525 | 100 | 3.08 | 12.46 | 20.10 |
| classical tensor network | 2,310,327,176 | 30 | 1.72 | 6.30 | 9.84 |

The classical baseline ran 30 epochs rather than 100. Its validation R@1 over
the last four epochs was 0.065, 0.067, 0.068 and 0.065 while its training loss
continued to fall from 2.450 to 2.319, so it had stopped improving on retrieval
before training stopped. Its per-epoch cost is roughly 33 minutes against 4 to
20 minutes for the quantum runs.

Full tables are in [`report/summary.md`](report/summary.md). Every measurement,
including intermediate checkpoints, is in
[`report/test_results.csv`](report/test_results.csv).

## Findings

**Accuracy rises with parameter count, and depth and width are not
interchangeable.** Fitting in parameters directly gives an exponent of 0.51
(95% bootstrap 0.31 to 0.69) for the width series at two layers, 0.58 (0.33 to
0.77) for depth at two qubits per type, and 0.37 (0.22 to 0.60) for depth at
three. `2 qubits, 5 layers` beats `3 qubits, 4 layers` while being 10% smaller.
The intervals overlap, so these are comparisons between specific configurations
rather than a fitted law. The range spans 0.67 decades of parameter count.

**Loss concentration is absent over the widths tested.** Resampling the text
tower 10,000 times without training, the variance of the loss changes by a
factor of 1.05 as the mean word register grows from 2.9 to 11.2 qubits. A
barren plateau predicts a factor of 304 over the same range. The relevant width
is the per-word register, not the caption total, because the resampled
parameters belong to individual word circuits.

<p align="center">
  <img src="report/figures/F5_loss_spread.png" width="42%">
  <img src="report/figures/F4_grammar_ablation.png" width="57%">
</p>

**Removing the grammar does not reduce retrieval accuracy.** Replacing the
parse with a commutative product, which makes word order invisible, scores 6.44
R@1 at 20,000 training images against 0.92 for the grammar model on the same
data. It uses 1,329,967 parameters against the grammar model's 479,353, so it
is larger rather than more efficient. The difference between the two appears on
word order: at 3,650 ARO pairs the standard error is 0.0083, and the flattened
variant sits 22.8 and 25.2 standard errors below chance on attributes and
relations, meaning it prefers the corrupted caption. The grammar model is 2.9
standard errors below chance on attributes and 4.9 above on relations.

**No ablation improved on the baseline.** Fourteen changes to the loss, the
image encoder, vocabulary sharing and the optimiser were tested at 10,000
images. All scored below the unmodified configuration. The three terms that
depart from the published loss (an arcsin warp, a purity penalty and label
smoothing) each contribute, the purity term most.

## Executors

The original executor passes every gate to `opt_einsum` separately, roughly 284
operands per caption. Two rewrites replace it, each checked against the
previous one.

`src/executors/compact_exec.py` contracts each word's gates into a single
tensor before the sentence network. Operands per caption drop to about 10, and
epoch time on full MSCOCO from 185 minutes to 13.

`src/executors/ring_sentence.py` represents a word as tensor-ring cores of bond
dimension `2^L` rather than a `2^N` statevector. A CNOT ring is a running XOR,
so the representation is exact with no truncation. At 21 wires this is 672
complex numbers rather than 2,097,152, and it is what makes the wider circuits
tractable.

Forward values agree to 1e-7, gradients to 1e-5, and training losses match to
four decimal places over 47 matched epochs. The checks are in
[`scripts/verify/`](scripts/verify).

## Fixes to the upstream code

Seven, of which four blocked training. The contrastive loss did not normalise
its embeddings, so image magnitude acted as a per-sample temperature. The
optimiser walked a million separate tensors per step. Contraction paths were
searched with the batch axis attached. The benchmark registered test vocabulary
before loading weights, so it scored a randomly initialised model. All are
collected in
[`scripts/patches/apply_fixes.py`](scripts/patches/apply_fixes.py).

## Reproducing

From the QCLIP root, with `report/` already populated:

```bash
python scripts/report/compile_results.py   # logs and results -> report/*.csv
python scripts/report/figures.py           # csv -> report/figures/*.pdf
```

Training and benchmarking run through `scripts/jobs/`, with the matching YAML
in `configs/`:

```bash
sbatch scripts/jobs/train.sh vqcfull_n2l5
sbatch scripts/jobs/bench.sh vqcfull_n2l5
```

## Layout

```
src/executors/     three ways of evaluating the circuits
src/training/      multi-positive dataset and loss
scripts/patches/   fixes and features applied to the QCLIP fork
scripts/analysis/  diagnostics that do not train anything
scripts/verify/    equivalence checks between executors
scripts/report/    result harvesting and figures
scripts/jobs/      Slurm submission scripts
configs/           one YAML per experiment
report/            compiled tables and figures
```

## Limitations

Each configuration was run once, so differences below about 0.3 in R@1 fall
within expected seed variance. No error bars are available on the retrieval
numbers.

The classical baseline trained for 30 epochs against the quantum models' 100,
as noted above.

Checkpoints were selected on a 1,000-image subset of the test set. The reported
figures use the final checkpoint, which does not depend on that selection.
Configurations were compared on the full test set during development, so the
choice of which configurations to report is not independent of the test set.

Published MSCOCO retrieval numbers use the Karpathy split; these use `val2017`.
Zero-shot CLIP scores 48.4 R@1 on this split against a published 50.0, which
gives a rough sense of the offset.

The parameter counts for the classical baseline vary with dataset size, because
the tensor-network encoder allocates per-vocabulary tensors: 480M at 5,000
images, 660M at 10,000, 940M at 20,000 and 2.31B on the full set.

## Citation

The model and codebase this builds on:

```
@inproceedings{limbackstokin2026meaning,
  title={Meaning Representations as Variational Quantum Circuits},
  author={Limb\"{a}ck-Stokin, Tilen G. and Birdavade, Tanishka A.
          and Lo, Kin Ian and Sadrzadeh, Mehrnoosh},
  booktitle={LREC},
  year={2026}
}
```

The classical tensor-network encoder used as a baseline is DisCoCLIP,
[Lo et al. (2025)](https://arxiv.org/abs/2509.21287).
