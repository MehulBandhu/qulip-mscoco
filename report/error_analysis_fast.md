# Error analysis: fast

24,909 captions over 5,000 images.

## By caption length (words)

| range | captions | top 1 (%) | top 10 (%) |
|---|---|---|---|
| (6.0, 8.0] | 4,265 | 0.75 | 6.07 |
| (8.0, 10.0] | 10,873 | 0.64 | 5.59 |
| (10.0, 12.0] | 6,259 | 0.61 | 4.67 |
| (12.0, 15.0] | 2,718 | 0.40 | 3.24 |
| (15.0, 100.0] | 794 | 0.38 | 2.02 |

## By circuit size (parameterised operands)

| range | captions | top 1 (%) | top 10 (%) |
|---|---|---|---|
| (100.0, 200.0] | 19,738 | 0.69 | 5.47 |
| (200.0, 300.0] | 4,944 | 0.36 | 3.70 |
| (300.0, 400.0] | 196 | 0.00 | 0.51 |
| (400.0, 100000.0] | 31 | 0.00 | 0.00 |

## By the widest grammatical type in the caption

| range | captions | top 1 (%) | top 10 (%) |
|---|---|---|---|
| (1.0, 2.0] | 3 | 0.00 | 0.00 |
| (2.0, 3.0] | 6,880 | 0.44 | 4.07 |
| (3.0, 4.0] | 2,760 | 0.51 | 3.59 |
| (4.0, 100.0] | 15,266 | 0.72 | 5.79 |

## Captions containing words the model never saw

| | captions | top 1 (%) | top 10 (%) |
|---|---|---|---|
| all symbols seen | 24,027 | 0.64 | 5.23 |
| one or more unseen | 882 | 0.11 | 0.68 |

## Correlation with the rank of the true caption

Spearman, so a handful of very bad cases cannot dominate.

| feature | correlation |
|---|---|
| words | +0.105 |
| operands | +0.091 |
| distinct symbols | +0.099 |
| unseen symbols | +0.161 |
| max type arity | -0.036 |

A positive value means the caption is ranked worse as that feature grows.

