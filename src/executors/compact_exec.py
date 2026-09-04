"""Compact executor: contract each word first, then the grammar.

The current path hands opt_einsum every gate of every word - about 269 operands
per caption - and pays Python dispatch on all of them. But a word's gates never
touch another word until its output wires reach the grammatical contraction, so
each word can be collapsed into a single tensor first. Words of the same arity
have identical circuit structure, so they all simulate together in a handful of
batched operations.

Contraction is associative, so this is the same model with the same parameters
and the same gradients. Only the evaluation order changes. Operands per caption
drop from ~269 to ~14.

Nothing here is wired into the model yet. Run it to check the compact path
reproduces the current one on real captions, then integrate:

    python compact_exec.py --n 32
"""
from __future__ import annotations

import argparse
import pickle
import time
from collections import defaultdict

import torch
from opt_einsum import contract

from modules.compilation.quantum.gates import Rz, Ry
from modules.utils.tensor_ops import interleaved2einsum


# --------------------------------------------------------------------------
# compilation

def compact_recipe(ansatz, tn):
    """One operand per word instead of one per gate.

    Mirrors BaseAnsatz.tn2ansatz exactly, except that a word contributes a
    single tensor carrying its N output wires rather than a chain of gates.
    The wire bookkeeping - fresh chars per word, then rewriting indices when a
    CCG index is already bound - is unchanged, so the resulting network is the
    same one the gate-level path builds, just already contracted per word.
    """
    ansatz.reset_char()
    ccg_map, input_indices, words = {}, [], []

    for word, idx_arr, type_arr in tn:
        arity = sum(ansatz.obmap.get(t, 1) for t in type_arr)
        out_wires = [ansatz.get_char() for _ in range(arity)]
        input_indices.append(out_wires)
        words.append({
            "symbol": f"{word}__{'@'.join(type_arr)}",
            "arity": arity,
        })

        i = 0
        for idx, typ in zip(idx_arr, type_arr):
            n = ansatz.obmap.get(typ, 1)
            if idx not in ccg_map:
                ccg_map[idx] = out_wires[i:i + n]
            else:
                swap = dict(zip(out_wires[i:i + n], ccg_map[idx]))
                input_indices = [[swap.get(w, w) for w in sub]
                                 for sub in input_indices]
            i += n

    return ansatz.gen_einsum_expr(input_indices, ccg_map), words


def angle_names(symbol: str, arity: int, layers: int):
    """The gate names this word's rotations carry, ordered [layer][wire][gate].

    CustomV5Ansatz numbers rotations with op_idx running over wires then the
    three gates, so wire i gate g is op_idx = 3*i + g. The CNOT ring also
    advances op_idx but its gates are unnamed, and it comes after the
    rotations, so it does not disturb this.
    """
    return [[[f"{symbol}_l{l}_{3 * i + g}" for g in range(3)]
             for i in range(arity)]
            for l in range(layers)]


# --------------------------------------------------------------------------
# execution

def simulate_words(angles: torch.Tensor, arity: int, layers: int):
    """All words of one arity at once. angles is [words, layers, arity, 3].

    Gate tensors carry indices (in, out), so applying one is state @ G. N
    Hadamards on |0> is |+> on every wire, so we start there and skip them.
    """
    n_words = angles.shape[0]
    state = torch.full((n_words, 2 ** arity), 2.0 ** (-arity / 2),
                       dtype=torch.complex64, device=angles.device)
    state = state.reshape(n_words, *([2] * arity))

    for l in range(layers):
        for w in range(arity):
            m = (Rz(angles[:, l, w, 0]) @ Ry(angles[:, l, w, 1])
                 @ Rz(angles[:, l, w, 2]))
            state = torch.movedim(state, w + 1, -1)
            flat = torch.einsum('wki,wij->wkj', state.reshape(n_words, -1, 2), m)
            state = torch.movedim(flat.reshape(state.shape), -1, w + 1)

        if arity > 1:
            for c in range(arity):
                t = (c + 1) % arity
                # CNOT is a permutation: where the control is 1, flip the target.
                state = torch.movedim(state, (c + 1, t + 1), (-2, -1))
                state = torch.stack([state[..., 0, :],
                                     state[..., 1, :].flip(-1)], dim=-2)
                state = torch.movedim(state, (-2, -1), (c + 1, t + 1))

    return state


def forward_compact(recipes, thetas, sym2param, unk_index, layers):
    """Word states for a whole batch, then the grammar contraction per caption.

    recipes is a list of (grammar_einsum, words) from compact_recipe.
    """
    # Gather every word in the batch, bucketed by arity.
    buckets = defaultdict(list)
    for r, (_, words) in enumerate(recipes):
        for w, spec in enumerate(words):
            buckets[spec["arity"]].append((r, w, spec))

    states = {}
    for arity, entries in buckets.items():
        idx = torch.tensor([
            [[[sym2param.get(name, unk_index) for name in gates]
              for gates in wires]
             for wires in angle_names(spec["symbol"], arity, layers)]
            for _, _, spec in entries
        ])
        batch = simulate_words(thetas[idx], arity, layers)
        for (r, w, _), s in zip(entries, batch):
            states[(r, w)] = s

    out = []
    for r, (expr, words) in enumerate(recipes):
        out.append(contract(expr, *[states[(r, w)] for w in range(len(words))]))
    return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="captions to check")
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    from modules.compilation.quantum.ansatz import CustomV5Ansatz
    from modules.models.text.einsum_quantum import VQCModel

    df = pickle.load(open("data/mscoco/processed/train_10k.pkl", "rb"))
    diagrams = [d[0] for d in df["captions_diagram"][: args.n]]
    obmap = {"n": 1, "s": 1, "p": 1, "out": 9}

    ansatz = CustomV5Ansatz(obmap=obmap, layers=args.layers)
    t = time.time()
    old = [ansatz(tn) for tn in diagrams]
    t_old_compile = time.time() - t

    t = time.time()
    new = [compact_recipe(ansatz, tn) for tn in diagrams]
    t_new_compile = time.time() - t

    ops_old = sum(len(a) for _, a in old) / len(old)
    ops_new = sum(len(w) for _, w in new) / len(new)
    print(f"operands per caption: {ops_old:.1f} -> {ops_new:.1f} "
          f"({ops_old / ops_new:.1f}x fewer)")
    print(f"compile: {t_old_compile:.2f}s -> {t_new_compile:.2f}s")

    model = VQCModel(out_q=obmap["out"])
    model.from_symbols([a for _, a in old], id_init=True)
    thetas = model.params.detach()

    t = time.time()
    want = model(old)
    t_old = time.time() - t

    t = time.time()
    got = forward_compact(new, thetas, model.sym2param,
                          model.unk_param_index, args.layers)
    t_new = time.time() - t

    print(f"\nforward: {t_old:.2f}s -> {t_new:.2f}s ({t_old / t_new:.1f}x)")

    worst = 0.0
    for i, g in enumerate(got):
        worst = max(worst, (want[i] - g).abs().max().item())
    print(f"max abs difference over {len(got)} captions: {worst:.2e}")
    print("MATCH" if worst < 1e-4 else "MISMATCH — do not integrate")


if __name__ == "__main__":
    main()
