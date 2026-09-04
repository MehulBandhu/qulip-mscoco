Two scripts, both taking a configuration name from `configs/` without the
`.yaml` extension.

```bash
sbatch scripts/jobs/train.sh vqcfull_n2l5      # train, then benchmark
sbatch scripts/jobs/bench.sh vqcfull_n2l5      # benchmark an existing run
```

Memory is requested by ansatz depth, since the tensor-ring bond dimension is
2^L and the evaluation pass holds all 24,909 test captions. A second argument
overrides it: `sbatch scripts/jobs/train.sh vqcfull_n4l4 900`.

Benchmarks are independent, so one job per configuration finishes the set in
the time a single one takes.

`superseded/` holds the per-run scripts these replaced.
