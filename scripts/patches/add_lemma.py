"""Key word angles on the lemma rather than the surface form.

"dog" and "dogs" currently learn entirely separate circuits, as do hold, holds,
holding and held. With 175,434 of ~339,000 symbols appearing exactly once and
42% of test symbols unseen, that duplication is where the model starves.

This changes only the lexical key. The CCG type, the wiring, the contraction and
the output qubits are untouched, and no classical network is added - which is the
difference from the angle generator, where a shared MLP replaced the whole table
and lost the memorisation that frequent words benefit from. Here frequent words
still learn independently of unrelated words; only inflections of the same stem
are pooled.

The part of speech comes from the CCG type rather than a tagger, since the parse
has already committed to it: a bare 'n' is a noun, a type mentioning 's' is
verbal, everything else is left alone.

    python -m scripts.add_lemma          # apply
    python -m scripts.add_lemma --check  # report only

Enable with `text.lemmatise: true`.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HELPER = '''
_LEMMA_CACHE: dict = {}
_LEMMATISER = None


def _lemma_pos(ccg_type: str) -> str:
    """Part of speech implied by the CCG type. WordNet wants 'n' or 'v'."""
    if not ccg_type or ccg_type == "n":
        return "n"
    return "v" if "s" in ccg_type else "n"


def _lemmatise(word: str, ccg_type: str) -> str:
    """Surface form to lemma, cached because the same words recur constantly.

    Falls back to the original word if WordNet is unavailable, so a missing
    corpus degrades to current behaviour rather than crashing a long run.
    """
    global _LEMMATISER
    key = (word, _lemma_pos(ccg_type))
    if key in _LEMMA_CACHE:
        return _LEMMA_CACHE[key]

    if _LEMMATISER is None:
        try:
            from nltk.stem import WordNetLemmatizer
            _LEMMATISER = WordNetLemmatizer()
            _LEMMATISER.lemmatize("dogs", "n")   # forces the corpus load
        except Exception as exc:
            print(f"  lemmatiser unavailable ({exc}); keeping surface forms")
            _LEMMATISER = False

    out = word if _LEMMATISER is False else _LEMMATISER.lemmatize(word, key[1])
    _LEMMA_CACHE[key] = out
    return out
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    p = REPO / "modules/compilation/quantum/ansatz.py"
    s = p.read_text()
    if "_lemmatise" in s:
        print("ansatz.py already patched")
    else:
        target = """            base_symbol = f"{word}__{'@'.join(type_arr)}\""""
        n = s.count(target)
        if n == 0:
            print("could not find the symbol construction; paste:")
            print("  grep -n 'base_symbol' modules/compilation/quantum/ansatz.py")
            sys.exit(1)
        print(f"  found {n} symbol construction site(s)")

        s = s.replace(target, """            _t = '@'.join(type_arr)
            _w = _lemmatise(word, _t) if getattr(self, 'lemmatise', False) else word
            base_symbol = f"{_w}__{_t}\"""")

        anchor = "class CustomV5Ansatz"
        if anchor not in s:
            anchor = "class "
        idx = s.index(anchor)
        s = s[:idx] + HELPER.strip() + "\n\n\n" + s[idx:]

        if not args.check:
            p.write_text(s)
        ast.parse(s)
        print("  + ansatz.py: lemmatised symbol keys")

    p = REPO / "modules/utils/factory.py"
    s = p.read_text()
    if "lemmatise" in s:
        print("factory.py already patched")
    else:
        anchor = "        ansatz.positional = ansatz_positional"
        if anchor not in s:
            print("could not find the ansatz flag block; paste:")
            print("  grep -n 'ansatz.positional' modules/utils/factory.py")
            sys.exit(1)
        s = s.replace(anchor, anchor +
                      "\n        ansatz.lemmatise = compiler.get('lemmatise', False)")
        if not args.check:
            p.write_text(s)
        ast.parse(s)
        print("  + factory.py: lemmatise wired")

    if not args.check:
        print("\nboth files parse")


if __name__ == "__main__":
    main()
