import pickle, collections
from modules.compilation.quantum.ansatz import CustomV5Ansatz

df = pickle.load(open('data/mscoco/processed/train_full.pkl','rb'))
sample = [d[0] for d in df['captions_diagram'][:3000]]

for n in (1, 2):
    a = CustomV5Ansatz(obmap={'n':n,'s':n,'p':n,'out':9}, layers=2)
    ar = collections.Counter()
    for tn in sample:
        for _, spec in [a.tn2ansatz_compact(tn)][0:1]:
            for w in spec:
                ar[w['arity']] += 1
    worst = max(ar)
    total_mb = sum(c * (2 ** k) * 8 for k, c in ar.items()) / 1e6 / len(sample) * 256
    print(f"n={n}: max arity {worst} = 2^{worst} = {2**worst:,} amplitudes")
    print(f"   distribution: {sorted(ar.items())[:6]} ... {sorted(ar.items())[-3:]}")
    print(f"   state memory for a 256-caption batch: {total_mb:,.0f} MB\n")
