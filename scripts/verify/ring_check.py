import pickle, torch
from modules.compilation.quantum.ansatz import CustomV5Ansatz
from modules.models.text.einsum_quantum import VQCModel

df = pickle.load(open('data/mscoco/processed/train_10k.pkl','rb')).head(16)
for q in (1, 3):
    a = CustomV5Ansatz(obmap={'n':q,'s':q,'p':q,'out':9}, layers=2)
    dense = a.compile_dataset(df.copy(), compact=True)
    ring = a.compile_dataset(df.copy(), ring=True)
    col = [c for c in dense.columns if c.endswith('_symbols')][0]
    e = [c for c in dense.columns if c.endswith('_einsum')][0]

    m1 = VQCModel(out_q=9); m1.from_symbols(dense[col].tolist(), id_init=True)
    m2 = VQCModel(out_q=9); m2.from_symbols(ring[col].tolist(), id_init=True)
    assert set(m1.symbols) == set(m2.symbols), "different parameter sets"
    m2.params.data = m1.params.data.clone()

    r1 = list(zip(dense[e].apply(lambda x: x[0]), dense[col].apply(lambda x: x[0])))
    r2 = list(zip(ring[e].apply(lambda x: x[0]), ring[col].apply(lambda x: x[0])))
    d = (m1(r1) - m2(r2)).abs().max().item()
    print(f"  n={q}: {len(m1.symbols)} symbols both, max diff {d:.2e} "
          f"{'MATCH' if d < 1e-4 else 'MISMATCH'}")
