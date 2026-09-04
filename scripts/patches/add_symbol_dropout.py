"""Train the role fallbacks by occasionally hiding words from the model.

The fallback angles added for unseen symbols never received a gradient: during
training every symbol is registered, so the fallback branch is dead code and the
angles stay at their initial values. Measured directly - 95% of table slots
moved over a run, 0% of fallback slots, max delta exactly zero.

Symbol dropout fixes that. Each forward pass picks a fraction of the vocabulary
and routes those words through their role fallback instead of their own angles.
The fallbacks then get gradient from ordinary words, and they are trained on
exactly the job they do at test time: standing in for a word the model has not
seen.

The choice is per word rather than per gate, so a dropped word loses all of its
angles at once, which is what an unseen word actually looks like. It is made by
hashing the word against a salt redrawn each forward pass, so it stays
consistent across that word's gates within a batch without needing to build and
pass a mask around.

    python scripts/patches/add_symbol_dropout.py
    python scripts/patches/add_symbol_dropout.py --check

Enable with `text.symbol_dropout: 0.1` in the config. Needs oov_fallback on,
since it routes through the role slots.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

OLD_LOOP = """                            if gate['name'] in self.sym2param:
                                param_indices.append(self.sym2param[gate['name']])
                            elif self.role_fallback is not None:
                                # Unseen word: use the learned circuit for its
                                # grammatical role rather than a shared zero.
                                role = _split_symbol(gate['name'])[1]
                                param_indices.append(
                                    self.role2slot.get(role, self.unk_param_index))
                            else:
                                param_indices.append(self.unk_param_index)"""

NEW_LOOP = """                            known = gate['name'] in self.sym2param
                            if known and self._drop_word(gate['name']):
                                # Hidden on purpose this pass, so the role
                                # fallback gets gradient from a word we do know.
                                known = False
                            if known:
                                param_indices.append(self.sym2param[gate['name']])
                            elif self.role_fallback is not None:
                                # Unseen (or hidden) word: use the learned
                                # circuit for its grammatical role rather than a
                                # shared zero.
                                role = _split_symbol(gate['name'])[1]
                                param_indices.append(
                                    self.role2slot.get(role, self.unk_param_index))
                            else:
                                param_indices.append(self.unk_param_index)"""

HELPER = '''
    def _drop_word(self, symbol: str) -> bool:
        """Whether to hide this word's own angles on this pass.

        Decided by hashing the word against a salt that changes every forward,
        so every gate of a word agrees within a batch and the choice is fresh
        between batches. Training only, and only with a fallback to route to.
        """
        rate = getattr(self, "symbol_dropout", 0.0)
        if not rate or not self.training or self.role_fallback is None:
            return False
        word = symbol.partition("__")[0]
        return (hash((word, self._drop_salt)) % 10_000) < rate * 10_000
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    p = REPO / "modules/models/text/einsum_quantum.py"
    s = p.read_text()
    if "_drop_word" in s:
        print("einsum_quantum.py already patched")
    else:
        assert s.count(OLD_LOOP) == 1, "param_indices loop not found"
        s = s.replace(OLD_LOOP, NEW_LOOP)

        anchor = "    def _angles(self):"
        assert s.count(anchor) == 1, "_angles not found"
        s = s.replace(anchor, HELPER.rstrip() + "\n\n" + anchor)

        anchor = "        thetas = self._angles()\n        dev = thetas.device"
        assert s.count(anchor) == 1, "forward's thetas line not found"
        s = s.replace(anchor,
                      "        # Redrawn per pass so the dropout choice changes between batches.\n"
                      "        self._drop_salt = random.randrange(1 << 30)\n"
                      "        thetas = self._angles()\n        dev = thetas.device")

        if "\nimport random" not in s and not s.startswith("import random"):
            s = "import random\n" + s
        if not args.check:
            p.write_text(s)
        ast.parse(s)
        print("  + einsum_quantum.py: symbol dropout added")

    p = REPO / "modules/utils/factory.py"
    s = p.read_text()
    if "symbol_dropout" in s:
        print("factory.py already patched")
    else:
        anchor = "        text_model.angle_generator = _gen"
        if anchor not in s:
            print("  could not find the flag block in factory.py; paste:")
            print("  grep -n 'angle_generator' modules/utils/factory.py")
            sys.exit(1)
        s = s.replace(anchor, anchor +
                      "\n        text_model.symbol_dropout = compiler.get('symbol_dropout', 0.0)")
        if not args.check:
            p.write_text(s)
        ast.parse(s)
        print("  + factory.py: symbol_dropout wired")

    if not args.check:
        print("\nboth files parse")


if __name__ == "__main__":
    main()
