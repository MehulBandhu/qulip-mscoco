import pickle, collections
from modules.compilation.quantum.ansatz import CustomV5Ansatz

df = pickle.load(open('data/mscoco/processed/train_10k.pkl','rb'))
a = CustomV5Ansatz(obmap={'n':1,'s':1,'p':1,'out':9}, layers=2)

now, arity_key, prefix_key = set(), set(), set()
arities = collections.defaultdict(set)
for diags in df['captions_diagram'][:4000]:
    for g in a(diags[0])[1]:
        if not g['name']:
            continue
        word, _, rest = g['name'].partition('__')
        ccg, _, tail = rest.rpartition('_l')
        layer, _, op = tail.partition('_')
        wire, gate = divmod(int(op), 3)
        ar = ccg.count('@') + 1 if ccg else 1
        now.add(g['name'])
        arity_key.add(f"{word}__a{ar}_l{layer}_{op}")
        # wire index only: the same wire of the same word is the same angle
        # whether that word has two wires here and five somewhere else.
        prefix_key.add(f"{word}__w{wire}_l{layer}_{gate}")
        arities[word].add(ar)

print(f"symbols now         {len(now):,}")
print(f"keyed on arity      {len(arity_key):,}  ({len(now)/len(arity_key):.2f}x)")
print(f"keyed on wire index {len(prefix_key):,}  ({len(now)/len(prefix_key):.2f}x)")
multi = [w for w, s in arities.items() if len(s) > 1]
print(f"\nwords appearing at more than one arity: {len(multi):,} of {len(arities):,}")
for w in sorted(multi, key=lambda w: -len(arities[w]))[:6]:
    print(f"  {w:12} arities {sorted(arities[w])}")
