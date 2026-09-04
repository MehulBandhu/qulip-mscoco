"""Feed tensor-ring cores straight into the sentence contraction.

The prototype contracted the cores back to a dense statevector to check them,
which throws away the whole point. Here each word contributes N cores instead of
one dense tensor: every core carries [left_bond, grammar_wire, right_bond], the
bonds are private to the word and close into a ring, and the physical leg keeps
whatever wire label the parse gave it. opt_einsum then contracts grammar and
word structure together and never builds 2^N amplitudes for anything.

Verified against the dense compact path, which is itself verified against the
original gate-by-gate executor.

    python ring_sentence.py --n 16
"""
from __future__ import annotations

import argparse
import pickle
import time
from collections import defaultdict

import torch
from opt_einsum import contract

from modules.compilation.quantum.gates import Rz, Ry


def cnot_ring_mpo(first: bool, dtype=torch.complex64):
    """W[carry_in, x, y, carry_out] for one site of the ring.

    Site 0 differs: its gate fires last and uses the total parity, so it passes
    its own x along the bond while the others pass y.
    """
    w = torch.zeros(2, 2, 2, 2, dtype=dtype)
    for carry in (0, 1):
        for x in (0, 1):
            y = x ^ carry
            w[carry, x, y, x if first else y] = 1.0
    return w


def word_cores(angles: torch.Tensor, arity: int, dtype=torch.complex64):
    """[words, arity, D, 2, D] for every word of one arity at once."""
    n_words, layers = angles.shape[0], angles.shape[1]
    dev = angles.device
    cores = torch.full((n_words, arity, 1, 2, 1), 2.0 ** -0.5, dtype=dtype,
                       device=dev)
    ring = None
    if arity > 1:          # a one-wire word has no ring to apply
        ring = torch.stack([cnot_ring_mpo(True, dtype).to(dev)]
                           + [cnot_ring_mpo(False, dtype).to(dev)] * (arity - 1))

    for l in range(layers):
        rot = torch.stack([
            (Rz(angles[:, l, w, 0]) @ Ry(angles[:, l, w, 1])
             @ Rz(angles[:, l, w, 2])).to(dtype) for w in range(arity)], dim=1)
        cores = torch.einsum('wnaxb,wnxy->wnayb', cores, rot)
        if ring is not None:
            d = cores.shape[2]
            cores = torch.einsum('wnaxb,nmxyc->wnamybc', cores, ring)
            cores = cores.reshape(n_words, arity, d * 2, 2, d * 2)
    return cores


def ring_recipe(ansatz, tn):
    """One operand per WIRE of each word, rather than one per word.

    The grammar bookkeeping is unchanged - fresh physical wires per word, then
    rewriting indices when a CCG index is already bound. Bond characters come
    from the same generator so they can never collide with wire characters,
    which is why the blanket rewrite below stays safe.
    """
    ansatz.reset_char()
    ccg_map, input_indices, words = {}, [], []

    for word, idx_arr, type_arr in tn:
        arity = sum(ansatz.obmap.get(t, 1) for t in type_arr)
        out_wires = [ansatz.get_char() for _ in range(arity)]
        bonds = [ansatz.get_char() for _ in range(arity)]

        slots = []
        for i in range(arity):
            slots.append(len(input_indices))
            # A one-wire word has bond dimension 1 and no ring, so its two bond
            # legs are the same character - a trace over a length-1 axis.
            right = bonds[(i + 1) % arity] if arity > 1 else bonds[0]
            input_indices.append([bonds[i], out_wires[i], right])

        words.append({"symbol": f"{word}__{'@'.join(type_arr)}",
                      "arity": arity, "slots": slots})

        i = 0
        for idx, typ in zip(idx_arr, type_arr):
            n = ansatz.obmap.get(typ, 1)
            if idx not in ccg_map:
                ccg_map[idx] = out_wires[i:i + n]
            else:
                swap = dict(zip(out_wires[i:i + n], ccg_map[idx]))
                input_indices = [[swap.get(c, c) for c in sub]
                                 for sub in input_indices]
            i += n

    return ansatz.gen_einsum_expr(input_indices, ccg_map), words


def angle_names(symbol, arity, layers):
    return [[[f"{symbol}_l{l}_{3 * i + g}" for g in range(3)]
             for i in range(arity)] for l in range(layers)]


def forward_ring(recipes, thetas, sym2param, unk_index, layers):
    buckets = defaultdict(list)
    for r, (_, words) in enumerate(recipes):
        for spec in words:
            buckets[spec["arity"]].append((r, spec))

    placed = {}
    for arity, entries in buckets.items():
        idx = torch.tensor([[[[sym2param.get(g, unk_index) for g in wire]
                              for wire in layer]
                             for layer in angle_names(s["symbol"], arity, layers)]
                            for _, s in entries])
        cores = word_cores(thetas[idx], arity)
        for (r, spec), c in zip(entries, cores):
            for k, slot in enumerate(spec["slots"]):
                placed[(r, slot)] = c[k]

    out = []
    for r, (expr, words) in enumerate(recipes):
        n_ops = sum(len(w["slots"]) for w in words)
        out.append(contract(expr, *[placed[(r, i)] for i in range(n_ops)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--qubits", type=int, default=1)
    args = ap.parse_args()

    from modules.compilation.quantum.ansatz import CustomV5Ansatz
    from modules.models.text.einsum_quantum import VQCModel
    from compact_exec import compact_recipe, forward_compact

    df = pickle.load(open("data/mscoco/processed/train_10k.pkl", "rb"))
    diagrams = [d[0] for d in df["captions_diagram"][: args.n]]
    q = args.qubits
    ansatz = CustomV5Ansatz(obmap={'n': q, 's': q, 'p': q, 'out': 9},
                            layers=args.layers)

    old = [ansatz(tn) for tn in diagrams]
    dense = [compact_recipe(ansatz, tn) for tn in diagrams]
    rings = [ring_recipe(ansatz, tn) for tn in diagrams]

    ops_dense = sum(len(w) for _, w in dense) / len(dense)
    ops_ring = sum(sum(len(x["slots"]) for x in w) for _, w in rings) / len(rings)
    print(f"operands per caption: dense words {ops_dense:.1f}, ring cores {ops_ring:.1f}")

    model = VQCModel(out_q=9)
    model.from_symbols([a for _, a in old], id_init=True)
    thetas = model.params.detach()

    t = time.time()
    want = forward_compact(dense, thetas, model.sym2param,
                           model.unk_param_index, args.layers)
    t_dense = time.time() - t

    t = time.time()
    got = forward_ring(rings, thetas, model.sym2param,
                       model.unk_param_index, args.layers)
    t_ring = time.time() - t

    worst = max((want[i] - got[i]).abs().max().item() for i in range(len(got)))
    print(f"dense {t_dense:.2f}s | ring {t_ring:.2f}s | {t_dense / t_ring:.2f}x")
    print(f"max difference over {len(got)} captions: {worst:.2e}")
    print("MATCH" if worst < 1e-4 else "MISMATCH")


if __name__ == "__main__":
    main()
