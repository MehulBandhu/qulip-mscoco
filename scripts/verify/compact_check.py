import pickle, torch
from modules.compilation.quantum.ansatz import CustomV5Ansatz
from modules.models.text.einsum_quantum import VQCModel

df = pickle.load(open('data/mscoco/processed/train_10k.pkl','rb')).head(16)
a = CustomV5Ansatz(obmap={'n':1,'s':1,'p':1,'out':9}, layers=2)

slow = a.compile_dataset(df.copy())
fast = a.compile_dataset(df.copy(), compact=True)
col = [c for c in slow.columns if c.endswith('_symbols')][0]

m1 = VQCModel(out_q=9); m1.from_symbols(slow[col].tolist(), id_init=True)
m2 = VQCModel(out_q=9); m2.from_symbols(fast[col].tolist(), id_init=True)
print(f"symbols registered: {len(m1.symbols)} slow vs {len(m2.symbols)} fast")
assert set(m1.symbols) == set(m2.symbols), "different parameter sets!"

m2.params.data = m1.params.data.clone()
e = [c for c in slow.columns if c.endswith('_einsum')][0]
r1 = list(zip(slow[e].apply(lambda x: x[0]), slow[col].apply(lambda x: x[0])))
r2 = list(zip(fast[e].apply(lambda x: x[0]), fast[col].apply(lambda x: x[0])))

d = (m1(r1) - m2(r2)).abs().max().item()
print(f"max abs difference: {d:.2e}")
print("MATCH" if d < 1e-4 else "MISMATCH")
