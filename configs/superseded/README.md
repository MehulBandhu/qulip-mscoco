Configurations that never produced a run.

Two kinds. The `o6`/`o7`/`o8` and `pos` variants failed on a shape mismatch in
the image projector, which assumed the output register was nine qubits because
2^9 happens to equal CLIP's embedding width; that is fixed, and the successors
are `vqcfull_n2l4q10` and `q11`. The rest were written for work that was
scoped out: transfer to SugarCrepe, an MLP baseline, and an earlier naming
scheme for the COCO runs.

Kept because they record what was attempted.
