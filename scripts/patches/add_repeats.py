"""Apply the CNOT ring more than once per layer.

Depth keeps helping, L=4 beats L=3 beats L=2, and the likely reason is reach
rather than expressiveness. The ring is 0->1->2->...->0, so information moves
one hop per layer; at 21 wires it takes 21 layers to cross a word and we have
two to four. Every extra layer is another hop.

Repeating the ring within a layer tests that directly, and cleanly: the CNOTs
carry no parameters, so L=2 with two rings has the SAME parameter count as L=2
with one, but the reach of L=4. If it matches L=4, reach is what depth was
buying. If L=4 still wins, the extra rotations matter too.

A log-depth entangler would mix faster still, but it would break the tensor-ring
representation, the bond dimension is 2^L only because the ring is
nearest-neighbour, with exactly one CNOT crossing any cut per layer. Long-range
gates put many CNOTs across the middle cut and the bond blows up. Repeating a
local ring keeps the bound at 2^(L*R).

    python scripts/patches/add_repeats.py --check
    python scripts/patches/add_repeats.py

Enable with `text.entangler_repeats: 2`.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def patch(rel, old, new, marker, check):
    p = REPO / rel
    s = p.read_text()
    if marker in s:
        return f"  . {rel}: already patched"
    if s.count(old) != 1:
        return f"  ! {rel}: expected 1 match, found {s.count(old)}"
    if not check:
        p.write_text(s.replace(old, new))
        ast.parse((REPO / rel).read_text())
    return f"  + {rel}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = []

    # --- the ring executor -------------------------------------------------
    out.append(patch(
        "modules/models/text/einsum_quantum.py",
        """            if ring is not None:
                d = cores.shape[2]
                cores = torch.einsum('wnaxb,nmxyc->wnamybc', cores, ring)
                cores = cores.reshape(n_words, arity, d * 2, 2, d * 2)""",
        """            if ring is not None:
                # Each application of the ring doubles the bond, so R rings per
                # layer gives bond dimension 2^(L*R) rather than 2^L.
                for _ in range(getattr(self, 'entangler_repeats', 1)):
                    d = cores.shape[2]
                    cores = torch.einsum('wnaxb,nmxyc->wnamybc', cores, ring)
                    cores = cores.reshape(n_words, arity, d * 2, 2, d * 2)""",
        "entangler_repeats", args.check))

    # --- the dense compact path, so both executors agree -------------------
    out.append(patch(
        "modules/models/text/einsum_quantum.py",
        """            if arity > 1:
                for c in range(arity):
                    t = (c + 1) % arity
                    # CNOT is a permutation: flip the target where control is 1.
                    state = torch.movedim(state, (c + 1, t + 1), (-2, -1))
                    state = torch.stack([state[..., 0, :],
                                         state[..., 1, :].flip(-1)], dim=-2)
                    state = torch.movedim(state, (-2, -1), (c + 1, t + 1))""",
        """            if arity > 1:
                for _ in range(getattr(self, 'entangler_repeats', 1)):
                    for c in range(arity):
                        t = (c + 1) % arity
                        # CNOT is a permutation: flip the target where the
                        # control is 1.
                        state = torch.movedim(state, (c + 1, t + 1), (-2, -1))
                        state = torch.stack([state[..., 0, :],
                                             state[..., 1, :].flip(-1)], dim=-2)
                        state = torch.movedim(state, (-2, -1), (c + 1, t + 1))""",
        "for _ in range(getattr(self, 'entangler_repeats', 1)):\n                    for c in range(arity):",
        args.check))

    # --- the gate-by-gate ansatz, so the reference path matches too --------
    p = REPO / "modules/compilation/quantum/ansatz.py"
    s = p.read_text()
    if "entangler_repeats" in s:
        out.append("  . ansatz.py: already patched")
    else:
        print("  ansatz.py CNOT block, for reference:")
        for i, line in enumerate(s.splitlines(), 1):
            if "'CX'" in line or "CX" in line and "op_type" in line:
                print(f"    {i}: {line.strip()[:78]}")
        out.append("  ? ansatz.py: patch by hand, see the lines above")

    # --- config plumbing ---------------------------------------------------
    out.append(patch(
        "modules/utils/factory.py",
        "        text_model.symbol_dropout = compiler.get('symbol_dropout', 0.0)",
        "        text_model.symbol_dropout = compiler.get('symbol_dropout', 0.0)\n"
        "        # Rings per layer. Adds no parameters, only reach.\n"
        "        text_model.entangler_repeats = compiler.get('entangler_repeats', 1)",
        "entangler_repeats = compiler", args.check))

    for line in out:
        print(line)
    if any(l.startswith("  !") for l in out):
        sys.exit(1)
    if not args.check:
        print("\nfiles parse")


if __name__ == "__main__":
    main()
