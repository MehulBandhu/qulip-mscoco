import pickle, collections
from modules.compilation.quantum.ansatz import CustomV5Ansatz

df = pickle.load(open('data/mscoco/processed/train_10k.pkl','rb'))
a = CustomV5Ansatz(obmap={'n':1,'s':1,'p':1,'out':9}, layers=2)

now, merged, per_word = set(), set(), collections.defaultdict(set)
for diags in df['captions_diagram'][:4000]:
    for g in a(diags[0])[1]:
        if not g['name']:
            continue
        word, _, rest = g['name'].partition('__')
        ccg, _, tail = rest.rpartition('_l')
        arity = ccg.count('@') + 1 if ccg else 1
        now.add(g['name'])
        merged.add(f"{word}__a{arity}_l{tail}")
        per_word[word].add(ccg)

print(f"symbols now      {len(now):,}")
print(f"after merging    {len(merged):,}   ({len(now)/len(merged):.2f}x fewer)")
multi = [w for w, t in per_word.items() if len(t) > 1]
print(f"words with more than one CCG type: {len(multi):,} of {len(per_word):,}")
for w in sorted(multi, key=lambda w: -len(per_word[w]))[:6]:
    print(f"  {w:14} {len(per_word[w])} types: {sorted(per_word[w])[:4]}")
