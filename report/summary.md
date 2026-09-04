# Results

Image-to-text recall on the full 5,000-image MSCOCO test set
(24,909 captions). Chance is 0.02%.

| configuration | text parameters | epochs | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|
| 2 qubits/type, 5 layers | 4,797,556 | 99 | 7.36 | 22.84 | 33.84 |
| 3 qubits/type, 4 layers | 5,363,041 | 99 | 6.46 | 21.54 | 33.26 |
| bag of words, 20k images | - | 36 | 6.44 | 20.58 | 31.00 |
| 2 qubits/type, 4 layers | 3,838,045 | 99 | 6.22 | 21.62 | 32.88 |
| 3 qubits/type, 3 layers | 4,022,281 | 99 | 5.44 | 17.58 | 27.82 |
| 4 qubits/type, 2 layers | - | 99 | 5.38 | 17.54 | 26.90 |
| 3 qubits/type, 2 layers | - | 99 | 4.98 | 17.50 | 27.46 |
| 2 qubits/type, 3 layers | 2,878,534 | 99 | 4.98 | 19.48 | 30.26 |
| 2 qubits/type, 2 layers | 1,919,023 | 99 | 4.36 | 14.24 | 22.84 |
| 2 qubits/type, 4 layers, 11 output qubits | 4,013,101 | 27 | 4.28 | 15.20 | 24.30 |
| 3 qubits/type, 2 layers, repeated entangler | 2,681,521 | 99 | 4.28 | 15.50 | 24.08 |
| 2 qubits/type, 4 layers, 10 output qubits | 3,925,573 | 53 | 4.08 | 16.20 | 25.60 |
| 2 qubits/type, 2 layers (reference executor) | 1,919,023 | 68 | 3.94 | 13.54 | 21.46 |
| vqcfull_n2l3 | - | 37 | 3.92 | 14.30 | 22.88 |
| 1 qubit/type, 2 layers, 2 captions per image | 1,156,525 | 99 | 3.40 | 12.24 | 20.48 |
| bag of words, 10k images | - | 29 | 3.24 | 11.14 | 18.36 |
| 1 qubit/type, 2 layers, 5 captions per image | 1,156,525 | 52 | 3.10 | 12.08 | 19.96 |
| 1 qubit/type, 2 layers | 1,156,525 | 99 | 3.08 | 12.46 | 20.10 |
| vqcfull_fast | - | 99 | 3.08 | 12.46 | 20.10 |
| vqcfull_k2 | - | 39 | 3.00 | 10.78 | 18.02 |
| 1 qubit/type, 2 layers (reference executor) | 1,156,525 | 59 | 2.96 | 11.44 | 18.88 |
| vqcfull_fastn2 | - | 32 | 2.92 | 12.28 | 19.52 |
| mscoco-vqcfull-long | - | 42 | 2.82 | 10.86 | 18.56 |
| mscoco-vqcfull-fast | - | 38 | 2.70 | 10.26 | 17.70 |
| vqcfull_long | - | 49 | 2.56 | 10.86 | 18.26 |
| vqcfull_n2 | - | 12 | 2.14 | 9.10 | 14.92 |
| vqcfull | - | 13 | 1.90 | 8.00 | 13.38 |
| bag of words (grammar removed) | 1,329,967 | 0 | 1.36 | 5.74 | 10.04 |
| classical, 20k images | - | 20 | 0.92 | 3.38 | 5.42 |
| vqc20k | - | 36 | 0.92 | 4.22 | 7.60 |
| classical bag of words, 10k images | - | 14 | 0.66 | 2.44 | 4.26 |
| image projection, 108 angles | - | 29 | 0.54 | 2.16 | 3.96 |
| vqc10k_v4 | - | 29 | 0.54 | 2.16 | 3.96 |
| image projection, 432 angles | - | 28 | 0.40 | 2.10 | 3.96 |
| vqc10k_v16 | - | 28 | 0.40 | 2.10 | 3.96 |
| image projection, 216 angles | - | 23 | 0.36 | 1.94 | 3.80 |
| vqc10k_v8 | - | 23 | 0.36 | 1.94 | 3.80 |
| n3 | 2,681,521 | 1 | 0.30 | 1.68 | 3.16 |
| image projection, 54 angles | - | 21 | 0.14 | 0.80 | 1.80 |
| vqc10k_v2 | - | 21 | 0.14 | 0.80 | 1.80 |
| vqc5k | - | 25 | 0.12 | 0.74 | 1.90 |

## ARO transfer, zero-shot from MSCOCO

Chance is 0.50.

| configuration | epochs | attribution | relation |
|---|---|---|---|
| aro_vqc20k | 36 | 0.4759 | 0.5403 |
| aro_vqc20k_spider | 36 | 0.3112 | 0.2915 |
| aro_vqcfull | 13 | 0.4876 | 0.5627 |
| q-aro-fast-10420430 | 99 | 0.4335 | 0.4101 |
| aro_vqc20k | 36 | 0.4759 | 0.5403 |
| aro_vqc20k_spider | 36 | 0.3112 | 0.2915 |
| aro_vqcfull | 13 | 0.4876 | 0.5627 |
| aro_vqcfull_fast | 99 | 0.4335 | 0.4101 |
