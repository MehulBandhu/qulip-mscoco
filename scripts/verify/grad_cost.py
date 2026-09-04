import pickle, time, torch
from modules.compilation.quantum.ansatz import CustomV5Ansatz
from modules.models.text.einsum_quantum import VQCModel

df = pickle.load(open('data/mscoco/processed/train_full.pkl','rb'))
diagrams = [d[0] for d in df['captions_diagram'][:256]]
for n in (1, 2):
    a = CustomV5Ansatz(obmap={'n':n,'s':n,'p':n,'out':9}, layers=2)
    old = [a(tn) for tn in diagrams]
    new = [a.tn2ansatz_compact(tn) for tn in diagrams]
    m = VQCModel(out_q=9); m.from_symbols([x for _,x in old], id_init=True)
    out = {}
    for name, r in (("gate", old), ("word", new)):
        t=time.time(); s=m(r); f=time.time()-t
        t=time.time(); s.abs().sum().backward(); b=time.time()-t; m.zero_grad()
        out[name]=(f,b)
    (fg,bg),(fw,bw) = out["gate"], out["word"]
    print(f"n={n}  gate fwd {fg:5.2f} bwd {bg:5.2f} | word fwd {fw:5.2f} bwd {bw:5.2f}"
          f" | fwd {fg/fw:5.1f}x  bwd {bg/bw:5.1f}x  total {(fg+bg)/(fw+bw):5.1f}x")
