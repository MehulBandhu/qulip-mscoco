import pickle, statistics, json
from modules.compilation.quantum.ansatz import CustomV5Ansatz

df = pickle.load(open('data/mscoco/processed/train_10k.pkl','rb'))
sample = [d[0] for d in df['captions_diagram'][:2000]]
out = {}
for n in (1, 2, 3, 4, 5):
    a = CustomV5Ansatz(obmap={'n':n,'s':n,'p':n,'out':9}, layers=2)
    widths, per_caption = [], []
    for tn in sample:
        _, words = a.tn2ansatz_ring(tn)
        w = [x['arity'] for x in words]
        widths.extend(w)
        per_caption.append(sum(w))
    out[n] = dict(mean_word=statistics.mean(widths),
                  median_word=statistics.median(widths),
                  max_word=max(widths),
                  mean_caption=statistics.mean(per_caption))
    print(f"  n={n}: word register mean {out[n]['mean_word']:.1f}, "
          f"median {out[n]['median_word']:.0f}, max {out[n]['max_word']}, "
          f"caption total {out[n]['mean_caption']:.1f}")
json.dump(out, open('report/arity_stats.json','w'), indent=1)
