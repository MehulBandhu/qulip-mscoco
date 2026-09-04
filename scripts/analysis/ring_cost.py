import pickle, math
from opt_einsum import contract_path
from modules.compilation.quantum.ansatz import CustomV5Ansatz

df = pickle.load(open('data/mscoco/processed/train_full.pkl','rb'))
for n in (3, 4, 5):
    a = CustomV5Ansatz(obmap={'n':n,'s':n,'p':n,'out':9}, layers=2)
    d = 2 ** a.layers
    worst_free = worst_cap = 0
    for tn in df['captions_diagram'][:40]:
        expr, words = a.tn2ansatz_ring(tn[0])
        shapes = []
        for w in words:
            for _ in w['slots']:
                shapes.append((d, 2, d) if w['arity'] > 1 else (1, 2, 1))
        for limit, tag in ((None, 'free'), (2 ** 24, 'cap')):
            _, info = contract_path(expr, *shapes, shapes=True,
                                    memory_limit=limit)
            if limit is None:
                worst_free = max(worst_free, info.largest_intermediate)
            else:
                worst_cap = max(worst_cap, info.largest_intermediate)
    print(f"n={n}: unbounded 2^{math.log2(worst_free):5.1f}  "
          f"capped 2^{math.log2(worst_cap):5.1f}")
