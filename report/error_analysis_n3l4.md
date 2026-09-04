# Error analysis: n3l4

24,909 captions over 5,000 images.

## By caption length (words)

| range | captions | top 1 (%) | top 10 (%) |
|---|---|---|---|
| (6.0, 8.0] | 4,265 | 1.52 | 10.41 |
| (8.0, 10.0] | 10,873 | 1.29 | 10.01 |
| (10.0, 12.0] | 6,259 | 1.10 | 8.96 |
| (12.0, 15.0] | 2,718 | 1.58 | 7.43 |
| (15.0, 100.0] | 794 | 0.76 | 4.53 |

## By circuit size (parameterised operands)

| range | captions | top 1 (%) | top 10 (%) |
|---|---|---|---|
| (400.0, 100000.0] | 24,909 | 1.30 | 9.36 |

## By the widest grammatical type in the caption

| range | captions | top 1 (%) | top 10 (%) |
|---|---|---|---|
| (1.0, 2.0] | 3 | 0.00 | 0.00 |
| (2.0, 3.0] | 6,880 | 0.76 | 6.82 |
| (3.0, 4.0] | 2,760 | 0.62 | 5.51 |
| (4.0, 100.0] | 15,266 | 1.66 | 11.20 |

## Captions containing words the model never saw

| | captions | top 1 (%) | top 10 (%) |
|---|---|---|---|
| all symbols seen | 24,027 | 1.34 | 9.63 |
| one or more unseen | 882 | 0.11 | 1.93 |

## Correlation with the rank of the true caption

Spearman, so a handful of very bad cases cannot dominate.

| feature | correlation |
|---|---|
| words | +0.070 |
| operands | +0.047 |
| distinct symbols | +0.064 |
| unseen symbols | +0.158 |
| max type arity | -0.094 |

A positive value means the caption is ranked worse as that feature grows.

