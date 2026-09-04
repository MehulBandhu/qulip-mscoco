# Scaling

## Recall against parameter count

Fitting R@1 = a N^b over 8 converged configurations:

- exponent b = 0.396 (95% bootstrap 0.337 to 0.457)
- R^2 in log space = 0.952

Every point is a single run, so this describes the configurations measured rather than estimating a population exponent. Repeat seeds would be needed for the latter.

## Depth against width

| | L=2 | L=3 | L=4 |
|---|---|---|---|
| n=2 | 4.36 | 4.98 | 6.22 |
| n=3 | 4.98 | 5.44 | 6.46 |
| n=4 | 5.38 | - | - |

- exponent in layers: 0.50, 0.37 (mean 0.43)
- exponent in qubits per type: 0.30 (mean 0.30)

## Recall against training-set size

At one qubit per type and two layers:

- exponent = 0.917 (95% bootstrap 0.394 to 2.544)
- R^2 in log space = 0.865

## Epochs to reach a training loss of 3.5

| configuration | parameters | epochs |
|---|---|---|
| neg | 479,353 | 0 |
| q-tn20k-10388753.out | 939,828,454 | 9 |
| q-tn10k-10389788.out@10k | 660,192,934 | 10 |
| q-tn10k_spider-10392852.out@10k | 660,192,934 | 10 |
| q-tnfull-10388754.out | 2,310,327,176 | 10 |
| q-tn5k-10389787.out | 480,006,760 | 11 |
| n=2, L=5 | 4,797,556 | 13 |
| spider | 1,329,967 | 13 |
| n=2, L=6 | 5,757,067 | 17 |
| n=3, L=4 | 5,363,041 | 17 |
| n=2, L=4 | 3,838,045 | 18 |
| n=2, L=4, 10 output qubits | 3,925,573 | 19 |
| n=2, L=4, 11 output qubits | 4,013,101 | 20 |
| spider@10k | 956,503 | 26 |
| n=2, L=3 | 2,878,534 | 27 |
| n=3, L=2 | 2,681,521 | 30 |
| n=3, L=3 | 4,022,281 | 32 |
| n=4, L=2 | 3,444,019 | 42 |
| n=3, L=2, ring x2 | 2,681,521 | 48 |
| n=2, L=2 (gate-by-gate) | 1,919,023 | 56 |
| n=2, L=2 | 1,919,023 | 57 |

- exponent in parameters: -0.170 (-0.226 to -0.123)

