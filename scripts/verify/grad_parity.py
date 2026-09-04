import sys; sys.path.insert(0, "src/executors")
import pickle, torch
from compact_exec import compact_recipe, forward_compact
from modules.compilation.quantum.ansatz import CustomV5Ansatz
from modules.models.text.einsum_quantum import VQCModel

df = pickle.load(open('data/mscoco/processed/train_10k.pkl','rb'))
diagrams = [d[0] for d in df['captions_diagram'][:16]]
obmap = {'n':1,'s':1,'p':1,'out':9}
a = CustomV5Ansatz(obmap=obmap, layers=2)
old = [a(tn) for tn in diagrams]
new = [compact_recipe(a, tn) for tn in diagrams]

m = VQCModel(out_q=9)
m.from_symbols([arr for _, arr in old], id_init=True)

# Same scalar objective through both paths, so the gradients are comparable.
m(old)[0].abs().sum().backward()
g_old = m.params.grad.clone(); m.zero_grad()

torch.stack([s.flatten() for s in forward_compact(
    new, m.params, m.sym2param, m.unk_param_index, 2)])[0].abs().sum().backward()
g_new = m.params.grad.clone()

d = (g_old - g_new).abs().max().item()
print(f"gradient max abs difference: {d:.2e}")
print(f"nonzero in each: {(g_old.abs()>1e-12).sum()} vs {(g_new.abs()>1e-12).sum()}")
print("MATCH" if d < 1e-5 else "MISMATCH do not integrate")
