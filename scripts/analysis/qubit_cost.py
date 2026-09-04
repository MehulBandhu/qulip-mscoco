import pickle, math
from opt_einsum import contract_path
from modules.compilation.quantum.ansatz import CustomV5Ansatz

df = pickle.load(open('data/mscoco/processed/train_full.pkl','rb'))
sample = [d[0] for d in df['captions_diagram'][:60]]

def shape_of(g):
    t = g['op_type']
    if t == '0': return (2,)
    if t in ('H', 'Rz', 'Rx', 'Ry'): return (2, 2)
    return (2, 2, 2, 2)          # CX and the controlled rotations

for n in (1, 2, 3):
    a = CustomV5Ansatz(obmap={'n': n, 's': n, 'p': n, 'out': 9}, layers=2)
    worst, flops = 0, []
    for tn in sample:
        expr, arr = a(tn)
        try:
            _, info = contract_path(expr, *[shape_of(g) for g in arr], shapes=True)
        except Exception:
            continue
        # Cost is dominated by the biggest tensor the contraction has to build,
        # so this is the number that decides whether a config is affordable.
        worst = max(worst, info.largest_intermediate)
        flops.append(info.opt_cost)
    print(f"n={n}: largest intermediate 2^{math.log2(worst):.1f} = {worst:,.0f} elements  "
          f"| mean flops {sum(flops)/len(flops):,.0f}")
