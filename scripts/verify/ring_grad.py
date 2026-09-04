import sys; sys.path.insert(0, "src/executors")
import pickle, torch
from ring_sentence import ring_recipe, forward_ring
from compact_exec import compact_recipe, forward_compact
from modules.compilation.quantum.ansatz import CustomV5Ansatz
from modules.models.text.einsum_quantum import VQCModel

df = pickle.load(open('data/mscoco/processed/train_10k.pkl','rb'))
diagrams = [d[0] for d in df['captions_diagram'][:12]]
a = CustomV5Ansatz(obmap={'n':3,'s':3,'p':3,'out':9}, layers=2)
old = [a(tn) for tn in diagrams]
m = VQCModel(out_q=9); m.from_symbols([x for _, x in old], id_init=True)

dense = [compact_recipe(a, tn) for tn in diagrams]
rings = [ring_recipe(a, tn) for tn in diagrams]

torch.stack([s.flatten() for s in forward_compact(
    dense, m.params, m.sym2param, m.unk_param_index, 2)]).abs().sum().backward()
g1 = m.params.grad.clone(); m.zero_grad()

torch.stack([s.flatten() for s in forward_ring(
    rings, m.params, m.sym2param, m.unk_param_index, 2)]).abs().sum().backward()
g2 = m.params.grad.clone()

d = (g1 - g2).abs().max().item()
print(f"gradient max difference: {d:.2e}")
print(f"nonzero: {(g1.abs()>1e-12).sum()} vs {(g2.abs()>1e-12).sum()}")
print("MATCH" if d < 1e-4 else "MISMATCH")
