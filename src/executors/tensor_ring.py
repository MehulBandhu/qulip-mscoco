"""Represent a word circuit as tensor-ring cores of bond dimension 2^L.

The dense path materialises the full statevector, which is fine at four wires
and hopeless at twenty-one. But the circuit has structure worth exploiting.

A CNOT ring is a running XOR: applying CNOT(c, c+1) around the ring sends
x -> y with y_i = x_i xor y_{i-1}, closing through the wraparound. Carrying one
parity bit along the chain expresses that exactly, so the ring is a
bond-dimension-2 matrix product operator. Single-qubit rotations do not touch
the bond at all. So L layers give bond dimension exactly

    D = 2^L

and at the L=2 used here, D = 4. A 21-wire word becomes 21 cores of shape
[4, 2, 4] - 672 complex numbers rather than 2,097,152 - with no truncation and
no approximation. Cost per word goes from O(L*N*2^N) to O(L*N*D^3), linear in N.

The cores can go straight into the sentence contraction: each core's physical
leg keeps its grammar wire label and its two bond legs are private to the word,
so opt_einsum contracts grammar and word structure together without ever
forming 2^N amplitudes.

    python tensor_ring.py     # verify against the dense path, and time it
"""
from __future__ import annotations

import time

import torch

from modules.compilation.quantum.gates import Rz, Ry


def cnot_ring_mpo(first: bool, dtype=torch.complex64) -> torch.Tensor:
    """One site of the CNOT-ring operator, W[carry_in, x, y, carry_out].

    The ring accumulates prefix parities P_i = x_0 xor ... xor x_i, leaving
    y_i = P_i on every site except the first. Site 0 is different because its
    gate fires last, using the total parity: y_0 = x_0 xor P_{N-1}. So site 0
    passes its own x along the bond while the others pass y. Making every site
    uniform instead forces P_{N-1} = 0 at the ring closure, which silently
    deletes every odd-parity amplitude.
    """
    w = torch.zeros(2, 2, 2, 2, dtype=dtype)
    for carry in (0, 1):
        for x in (0, 1):
            y = x ^ carry
            w[carry, x, y, x if first else y] = 1.0
    return w


def word_cores(angles: torch.Tensor, arity: int,
               dtype=torch.complex64) -> torch.Tensor:
    """Tensor-ring cores for every word of one arity at once.

    angles is [words, layers, arity, 3]; the result is
    [words, arity, D, 2, D] with the bonds forming a ring, so the word's
    amplitude for a bit string is the trace of the product of its cores.
    """
    n_words, layers = angles.shape[0], angles.shape[1]
    device = angles.device

    # |+> on every wire: bond dimension 1, amplitude 2^(-1/2) either way.
    cores = torch.full((n_words, arity, 1, 2, 1), 2.0 ** -0.5,
                       dtype=dtype, device=device)
    ring_first = cnot_ring_mpo(True, dtype).to(device)
    ring_rest = cnot_ring_mpo(False, dtype).to(device)
    ring = torch.stack([ring_first] + [ring_rest] * (arity - 1))

    for l in range(layers):
        # Rotations act on the physical leg only, so the bond is untouched.
        rot = torch.stack([
            (Rz(angles[:, l, w, 0]) @ Ry(angles[:, l, w, 1])
             @ Rz(angles[:, l, w, 2])).to(dtype)
            for w in range(arity)], dim=1)                    # [words, arity, 2, 2]
        cores = torch.einsum('wnaxb,wnxy->wnayb', cores, rot)

        # The ring doubles the bond: the state's own bond travels alongside the
        # parity carry, and the pair becomes the new bond.
        d = cores.shape[2]
        cores = torch.einsum('wnaxb,nmxyc->wnamybc', cores, ring)
        cores = cores.reshape(
            n_words, arity, d * 2, 2, d * 2)

    return cores


def cores_to_dense(cores: torch.Tensor) -> torch.Tensor:
    """Contract the ring back to a full statevector, for checking only."""
    n_words, arity = cores.shape[0], cores.shape[1]
    d = cores.shape[2]
    # Carry the open left bond along, then close it against the right at the end.
    acc = torch.eye(d, dtype=cores.dtype, device=cores.device)
    acc = acc.reshape(1, d, 1, d).expand(n_words, d, 1, d).clone()
    for n in range(arity):
        acc = torch.einsum('walb,wbxc->walxc', acc, cores[:, n])
        acc = acc.reshape(n_words, d, -1, acc.shape[-1])
    # Trace over the ring bond.
    return torch.einsum('waxa->wx', acc).reshape(
        n_words, *([2] * arity))


def dense_reference(angles, arity, dtype=torch.complex64):
    """The current implementation, to check against."""
    n_words = angles.shape[0]
    state = torch.full((n_words, 2 ** arity), 2.0 ** (-arity / 2), dtype=dtype)
    state = state.reshape(n_words, *([2] * arity))
    for l in range(angles.shape[1]):
        for w in range(arity):
            m = (Rz(angles[:, l, w, 0]) @ Ry(angles[:, l, w, 1])
                 @ Rz(angles[:, l, w, 2])).to(dtype)
            state = torch.movedim(state, w + 1, -1)
            flat = torch.einsum('wki,wij->wkj',
                                state.reshape(n_words, -1, 2), m)
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
    print(f"  {'arity':>6} {'dense (s)':>10} {'ring (s)':>9} {'numbers':>12} "
          f"{'vs dense':>9}  {'max diff':>10}")

    for arity in (2, 3, 4, 6, 8, 10, 12):
        angles = torch.rand(16, 2, arity, 3) * 2 - 1

        t = time.time(); want = dense_reference(angles, arity)
        t_dense = time.time() - t
        t = time.time(); cores = word_cores(angles, arity)
        t_ring = time.time() - t

        got = cores_to_dense(cores)
        err = (want - got).abs().max().item()
        per_word = cores.shape[1] * cores.shape[2] * 2 * cores.shape[4]
        flag = "" if err < 1e-5 else "   MISMATCH"
        print(f"  {arity:>6} {t_dense:>10.3f} {t_ring:>9.3f} {per_word:>12,} "
              f"{2 ** arity / per_word:>8.2f}x  {err:>10.2e}{flag}")

    print("\n  'numbers' is the cores' size; 'vs dense' is how much smaller")
    print("  than the 2^N statevector. The gap widens fast with arity.")


if __name__ == "__main__":
    main()
