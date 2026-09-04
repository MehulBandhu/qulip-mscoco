import pickle, collections
from modules.compilation.quantum.ansatz import CustomV5Ansatz

a = CustomV5Ansatz(obmap={'n':1,'s':1,'p':1,'out':9}, layers=2)
def syms(df, n=None):
    c = collections.Counter()
    for diags in (df['captions_diagram'][:n] if n else df['captions_diagram']):
        for d in diags:
            for g in a(d)[1]:
                if g['name']:
                    c[g['name']] += 1
    return c

train = syms(pickle.load(open('data/mscoco/processed/train_full.pkl','rb')))
test = syms(pickle.load(open('data/mscoco/processed/validation_20k.pkl','rb')))
unseen = sum(1 for s in test if s not in train)
once = sum(1 for v in train.values() if v == 1)
print(f"train symbols {len(train):,}, seen once {once:,} ({once/len(train):.1%})")
print(f"test symbols  {len(test):,}, unseen {unseen:,} ({unseen/len(test):.1%})")
