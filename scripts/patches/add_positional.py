"""Give spider mode a sense of word order.

Spider multiplies the word states together, and multiplication commutes, so it
scores "the dog bites the man" and "the man bites the dog" identically. That is
fine for retrieval, which is mostly lexical, but it is exactly what ARO and
SugarCrepe are built to catch.

The fix here applies an extra rotation to each word's state before the product,
indexed by the word's POSITION in the caption. Because R1(A)*R2(B) and
R1(B)*R2(A) differ, order now matters. The rotations are shared across all words
at a given position, so the cost is a few thousand parameters, and they run
alongside the word circuits rather than after them, so depth is unchanged. That
matters: depth is what made the grammar version untrainable.

    python -m scripts.add_positional          # apply
    python -m scripts.add_positional --check  # report only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def patch(rel: str, old: str, new: str, marker: str, check: bool) -> str:
    path = REPO / rel
    src = path.read_text()
    if marker in src:
        return f"  . {rel}: already patched"
    n = src.count(old)
    if n != 1:
        return f"  ! {rel}: expected 1 match, found {n}"
    if not check:
        path.write_text(src.replace(old, new))
    return f"  + {rel}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = []

    out.append(patch(
        "modules/compilation/quantum/ansatz.py",
        """            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            input_indices.extend(new_indices)
            tensor_arr.extend(new_tensors)

            ccg_map[idx] = current_wires""",
        """            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            input_indices.extend(new_indices)
            tensor_arr.extend(new_tensors)

            # Optional: a rotation that depends on where the word sits in the
            # caption. Without it the product over words commutes and the model
            # cannot tell "dog bites man" from "man bites dog". Parameters are
            # shared across every word at a given position, and the block runs
            # alongside the word circuits so depth is unchanged.
            if getattr(self, 'positional', False):
                slot = min(idx, getattr(self, 'max_positions', 24) - 1)
                current_wires, pos_indices, pos_tensors = self.ansatz(
                    current_wires, f"POSITION{slot}__pos")
                input_indices.extend(pos_indices)
                tensor_arr.extend(pos_tensors)

            ccg_map[idx] = current_wires""",
        "POSITION{slot}__pos", args.check))

    out.append(patch(
        "modules/utils/factory.py",
        "        print(f\"  text ansatz : {name}\")",
        """        print(f"  text ansatz : {name}")
        # Position-indexed rotations, only meaningful in spider mode where the
        # word states are combined by a commutative product.
        ansatz_positional = compiler.get('positional', False)""",
        "ansatz_positional = compiler.get", args.check))

    out.append(patch(
        "modules/utils/factory.py",
        "        ansatz = ansatze[name](obmap=obmap, layers=compiler['layers'])",
        """        ansatz = ansatze[name](obmap=obmap, layers=compiler['layers'])
        ansatz.positional = ansatz_positional
        ansatz.max_positions = compiler.get('max_positions', 24)
        if ansatz_positional:
            print(f"  positional  : on, {ansatz.max_positions} slots")""",
        "ansatz.positional = ansatz_positional", args.check))

    for line in out:
        print(line)

    if not args.check and not any(l.startswith("  !") for l in out):
        import ast
        for f in ("modules/compilation/quantum/ansatz.py", "modules/utils/factory.py"):
            ast.parse((REPO / f).read_text())
        print("\nboth files parse")
    sys.exit(1 if any(l.startswith("  !") for l in out) else 0)


if __name__ == "__main__":
    main()
