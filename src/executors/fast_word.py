"""Apply a CNOT ring as a single permutation of the statevector.
training code until it has been verified.

The CNOT ring is currently applied one gate at a time, and each gate does a
movedim, a stack and a flip, so a full pass over the statevector, N times per
layer. But a ring of CNOTs is a linear map over GF(2): every gate sends a basis
state to another basis state, so the whole ring is one permutation of the 2^N
amplitudes. Precompute it once per arity and the N passes become a single
gather.

That leaves the 3N rotation passes untouched, so the saving is roughly a
quarter. The real win for wide words is the tensor-ring representation, which
this file is the groundwork for: with L layers the CNOT bond dimension is 2^L,
so at L=2 a 21-wire word needs 21*4*2*4 = 672 numbers rather than 2,097,152.

    python fast_word.py            # verify and time against the current path
"""
from __future__ import annotations

import time

import torch

from modules.compilation.quantum.gates import Rz, Ry


_PERM_CACHE: dict[int, torch.Tensor] = {}


def cnot_ring_permutation(arity: int) -> torch.Tensor:
    """Index map for a full CNOT ring, as a gather over the 2^N amplitudes.

    Applying CNOT(c, c+1) puts the amplitude that was at (x with bit c+1
    flipped when bit c is set) into position x, a gather, not a scatter. The
    ring applies gates 0..N-1 in order, so composing the gathers means walking
    the gates backwards.
    """
    if arity in _PERM_CACHE:
        return _PERM_CACHE[arity]

    perm = torch.arange(1 << arity)
    for c in range(arity - 1, -1, -1):
        t = (c + 1) % arity
        bit_c = (perm >> (arity - 1 - c)) & 1
        perm = perm ^ (bit_c << (arity - 1 - t))
    _PERM_CACHE[arity] = perm
    return perm


def simulate_words_fast(angles: torch.Tensor, arity: int,
                        precision=torch.complex64) -> torch.Tensor:
    """All words of one arity at once. angles is [words, layers, arity, 3]."""
    n_words, layers = angles.shape[0], angles.shape[1]
    state = torch.full((n_words, 1 << arity), 2.0 ** (-arity / 2),
                       dtype=precision, device=angles.device)

    perm = None if arity == 1 else cnot_ring_permutation(arity).to(angles.device)

    for l in range(layers):
        state = state.reshape(n_words, *([2] * arity))
        for w in range(arity):
            m = (Rz(angles[:, l, w, 0]) @ Ry(angles[:, l, w, 1])
                 @ Rz(angles[:, l, w, 2])).to(precision)
            state = torch.movedim(state, w + 1, -1)
            flat = torch.einsum('wki,wij->wkj', state.reshape(n_words, -1, 2), m)
            state = torch.movedim(flat.reshape(state.shape), -1, w + 1)

        if perm is not None:
            # One gather for the whole ring instead of a pass per gate.
            state = state.reshape(n_words, -1)[:, perm]

    return state.reshape(n_words, *([2] * arity))


def simulate_words_reference(angles, arity, precision=torch.complex64):
    """The current implementation, kept here to check against."""
    n_words = angles.shape[0]
    state = torch.full((n_words, 2 ** arity), 2.0 ** (-arity / 2),
                       dtype=precision, device=angles.device)
    state = state.reshape(n_words, *([2] * arity))

    for l in range(angles.shape[1]):
        for w in range(arity):
            m = (Rz(angles[:, l, w, 0]) @ Ry(angles[:, l, w, 1])
                 @ Rz(angles[:, l, w, 2])).to(precision)
            state = torch.movedim(state, w + 1, -1)
            flat = torch.einsum('wki,wij->wkj', state.reshape(n_words, -1, 2), m)
            state = torch.movedim(flat.reshape(state.shape), -1, w + 1)

        if arity > 1:
            for c in range(arity):
                t = (c + 1) % arity
                state = torch.movedim(state, (c + 1, t + 1), (-2, -1))
                state = torch.stack([state[..., 0, :],
                                     state[..., 1, :].flip(-1)], dim=-2)
                state = torch.movedim(state, (-2, -1), (c + 1, t + 1))
    return state


def main():
    torch.manual_seed(0)
    print(f"  {'arity':>6} {'words':>7} {'ref (s)':>9} {'fast (s)':>9} "
          f"{'gain':>6}  {'max diff':>10}")

    for arity, n_words in ((2, 4000), (4, 2000), (8, 500), (12, 100),
                           (15, 30), (18, 8), (21, 2)):
        angles = torch.rand(n_words, 2, arity, 3) * 2 - 1
        if arity > 1:
            cnot_ring_permutation(arity)      # build it before timing

        t = time.time(); want = simulate_words_reference(angles, arity)
        t_ref = time.time() - t
        t = time.time(); got = simulate_words_fast(angles, arity)
        t_new = time.time() - t

        err = (want - got).abs().max().item()
        flag = "" if err < 1e-5 else "   MISMATCH"
        print(f"  {arity:>6} {n_words:>7} {t_ref:>9.3f} {t_new:>9.3f} "
              f"{t_ref / max(t_new, 1e-9):>6.2f}x  {err:>10.2e}{flag}")

    # Gradients have to match too, not just values.
    angles = (torch.rand(50, 2, 8, 3) * 2 - 1).requires_grad_(True)
    simulate_words_reference(angles, 8).abs().sum().backward()
    g_ref = angles.grad.clone(); angles.grad = None
    simulate_words_fast(angles, 8).abs().sum().backward()
    d = (g_ref - angles.grad).abs().max().item()
    print(f"\n  gradient max difference at arity 8: {d:.2e}"
          f"{'' if d < 1e-4 else '   MISMATCH'}")


if __name__ == "__main__":
    main()
