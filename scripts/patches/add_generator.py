"""Two ways to stop learning every (word, type, gate) angle independently.

Both are off by default and enabled from the config, so the baseline is
untouched.

  text.oov_fallback: true
      An unseen gate currently routes to one slot pinned at zero. Zero is not a
      neutral angle here - the circuit opens with Hadamards, so every unknown
      word ends up in the same all-plus state whether it is a noun or a
      transitive verb. This gives each grammatical role its own learnable
      fallback instead: roughly a few hundred extra angles covering the 42% of
      test symbols that currently collapse together.

  text.angle_generator: true
      Replaces the per-symbol table with
          theta = base[role] + MLP([word_embedding, role_embedding])
      so "dogs" can borrow from "dog" and an unseen word gets angles from its
      embedding rather than a fallback. Parameter count stops tracking
      vocabulary. The whole angle vector is generated once per forward pass with
      the same shape the table had, so the gather, the topology bucketing and
      the contraction are all unchanged.

    python -m scripts.add_generator
    python -m scripts.add_generator --check
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HELPERS = '''
def _split_symbol(symbol: str) -> tuple[str, str]:
    """'holds__n@out@n_l0_7' -> ('holds', 'n@out@n_l0_7'). The role is
    everything that is not the word: type, layer and gate index."""
    word, _, role = symbol.partition("__")
    return word.lower(), role


def _word_embeddings(words, dim=512):
    """Frozen CLIP token embeddings, averaged over sub-tokens for words CLIP
    splits up. Returns a (len(words), dim) tensor and never trains."""
    import clip
    model, _ = clip.load("ViT-B/32", device="cpu")
    table = model.token_embedding.weight.detach()
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    out = torch.zeros(len(words), table.shape[1])
    for i, w in enumerate(words):
        ids = tokenizer.encode(w)
        if ids:
            out[i] = table[torch.tensor(ids)].mean(0)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-9)

'''

INIT_TAIL = '''
        # --- optional: per-role fallback for unseen symbols ----------------
        self.role_fallback = None
        if getattr(self, "oov_fallback", False):
            roles = sorted({_split_symbol(s)[1] for s in self.symbols})
            self.role2slot = {r: num_symbols + 1 + i for i, r in enumerate(roles)}
            extra = torch.empty(len(roles))
            extra.uniform_(-0.01, 0.01) if id_init else extra.uniform_(-torch.pi/2, torch.pi/2)
            with torch.no_grad():
                self.params = nn.Parameter(torch.cat([self.params.detach(), extra]))
            self.params.register_hook(
                lambda g, i=num_symbols: g.index_fill(0, torch.tensor([i], device=g.device), 0.0))
            self.role_fallback = True
            print(f"  oov fallback: {len(roles)} grammatical roles")

        # --- optional: generate angles instead of storing them -------------
        self.generator = None
        if getattr(self, "angle_generator", False):
            words, roles = zip(*[_split_symbol(s) for s in self.symbols])
            vocab = sorted(set(words))
            role_list = sorted(set(roles))
            w2i = {w: i for i, w in enumerate(vocab)}
            r2i = {r: i for i, r in enumerate(role_list)}

            self.register_buffer("gen_word_emb", _word_embeddings(vocab))
            self.register_buffer("gen_word_idx",
                                 torch.tensor([w2i[w] for w in words]))
            self.register_buffer("gen_role_idx",
                                 torch.tensor([r2i[r] for r in roles]))

            role_dim, hidden = 32, 128
            self.gen_role_emb = nn.Embedding(len(role_list), role_dim)
            self.gen_base = nn.Parameter(torch.zeros(len(role_list)))
            self.generator = nn.Sequential(
                nn.Linear(self.gen_word_emb.shape[1] + role_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            # Start near identity like the table does, so the circuit begins
            # close to the same place and the comparison stays fair.
            nn.init.zeros_(self.generator[-1].bias)
            nn.init.normal_(self.generator[-1].weight, std=0.01)

            n = sum(p.numel() for p in [self.gen_base, *self.gen_role_emb.parameters(),
                                        *self.generator.parameters()])
            print(f"  angle generator: {len(vocab):,} words, {len(role_list)} roles, "
                  f"{n:,} parameters instead of {num_symbols:,}")

    def _angles(self):
        """The angle vector the contraction reads. Either the stored table or,
        with the generator on, one produced from word and role embeddings."""
        if self.generator is None:
            return self.params
        feats = torch.cat([self.gen_word_emb[self.gen_word_idx],
                           self.gen_role_emb(self.gen_role_idx)], dim=-1)
        generated = self.gen_base[self.gen_role_idx] + self.generator(feats).squeeze(-1)
        # Keep the trailing unknown slot so index arithmetic downstream is
        # unchanged; with the generator on nothing should route to it.
        return torch.cat([generated, self.params[-1:].detach() * 0])
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    done = []

    p = REPO / "modules/models/text/einsum_quantum.py"
    s = p.read_text()
    if "_split_symbol" in s:
        print("einsum_quantum.py already patched")
    else:
        anchor = "class VQCModel"
        assert s.count(anchor) >= 1, "VQCModel not found"
        s = s.replace(anchor, HELPERS.strip() + "\n\n\n" + anchor, 1)

        anchor = """        self.params.register_hook(
            lambda g, i=num_symbols: g.index_fill(0, torch.tensor([i], device=g.device), 0.0))"""
        assert s.count(anchor) == 1, "init_params tail not found"
        s = s.replace(anchor, anchor + "\n" + INIT_TAIL)

        anchor = "        thetas = self.params\n        dev = thetas.device"
        assert s.count(anchor) == 1, "forward's thetas line not found"
        s = s.replace(anchor, "        thetas = self._angles()\n        dev = thetas.device")

        anchor = """                            if gate['name'] in self.sym2param:
                                param_indices.append(self.sym2param[gate['name']])
                            else:
                                param_indices.append(self.unk_param_index)"""
        assert s.count(anchor) == 1, "param_indices loop not found"
        s = s.replace(anchor, """                            if gate['name'] in self.sym2param:
                                param_indices.append(self.sym2param[gate['name']])
                            elif self.role_fallback is not None:
                                # Unseen word: use the learned circuit for its
                                # grammatical role rather than a shared zero.
                                role = _split_symbol(gate['name'])[1]
                                param_indices.append(
                                    self.role2slot.get(role, self.unk_param_index))
                            else:
                                param_indices.append(self.unk_param_index)""")

        if not args.check:
            p.write_text(s)
        ast.parse(s)
        done.append("einsum_quantum.py: fallback and generator added")

    p = REPO / "modules/utils/factory.py"
    s = p.read_text()
    if "oov_fallback" in s:
        print("factory.py already patched")
    else:
        anchor = "        text_model = VQCModel("
        if anchor not in s:
            print("  could not find the VQCModel construction in factory.py;")
            print("  paste: grep -n 'VQCModel(' modules/utils/factory.py")
            sys.exit(1)
        s = s.replace(anchor,
                      "        # Read before the model registers symbols, since both options\n"
                      "        # change what init_params builds.\n"
                      "        _oov = compiler.get('oov_fallback', False)\n"
                      "        _gen = compiler.get('angle_generator', False)\n" + anchor, 1)
        line_end = s.index("\n", s.index(anchor))
        s = s[:line_end + 1] + (
            "        text_model.oov_fallback = _oov\n"
            "        text_model.angle_generator = _gen\n") + s[line_end + 1:]
        if not args.check:
            p.write_text(s)
        ast.parse(s)
        done.append("factory.py: config flags wired")

    for line in done:
        print(f"  + {line}")
    if done and not args.check:
        print("\nboth files parse")


if __name__ == "__main__":
    main()
