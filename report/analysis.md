# Analysis

Image-to-text recall on the full 5,000-image MSCOCO test set,
24,909 captions, chance 0.02%. One run per configuration.

## Parameters

R@1 grows as N^0.471 (95% bootstrap 0.331 to 0.557, R^2 0.946, 8 configurations).

## Depth against width

| | L=2 | L=3 | L=4 |
|---|---|---|---|
| n=1 | 3.08 | - | - |
| n=2 | 4.36 | 4.98 | 6.22 |
| n=3 | 4.98 | 5.44 | 6.46 |
| n=4 | 5.38 | - | - |

- exponent in layers: 0.43
- exponent in qubits per type: 0.41

Depth scales roughly 1.1 times better per parameter than width.

## Training dynamics

Gradient norms rise monotonically in every configuration, and the similarity gap between true pairs and the rest grows throughout. Neither shows the flattening a trainability problem would produce.

## Training-set size

- quantum tower, one qubit per type: R@1 ~ D^0.92
- classical tower: R@1 ~ D^0.36

The quantum tower has the steeper data exponent despite having 1998 times fewer parameters.

## Loss concentration at initialisation

Resampling the text tower 10,000 times without training, the variance of the loss is flat to within 4% while the mean circuit width grows 3.8 times (29.3 to 110.4 qubits per caption). There is no exponential concentration at any width tested, which is why wider and deeper circuits keep helping.

## Convergence speed

Epochs to reach a training loss of 3.5 fall as N^-1.12, so larger circuits are not merely better at convergence, they get there sooner.

## Ablations

Every modification tested at 10,000 images scored below the unmodified baseline. The three apparent deviations from the published loss (an arcsin warp, a purity penalty and label smoothing) all turn out to help, the purity term most of all.

## Execution

Contracting each word's circuit before the sentence network takes a caption from about 284 operands to 10, and is 14 times faster at one qubit per type. Representing a word as tensor-ring cores of bond dimension 2^L rather than a 2^N statevector is what keeps the wider circuits affordable. Both were checked against the original executor: forward values agree to 1e-7, gradients to 1e-5, and training losses match to four decimals over 47 epochs.

