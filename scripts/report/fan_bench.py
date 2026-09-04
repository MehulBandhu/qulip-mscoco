"""Submit every benchmark as its own job instead of one long queue.

A benchmark is 20-30 minutes and they are completely independent, so running
fourteen in sequence wastes most of a day for no reason. This writes one script
per checkpoint and submits them all; the whole set finishes in the time one of
them takes.

Memory is requested per configuration rather than uniformly - bond dimension is
2^L, so a two-layer run needs a fraction of what a four-layer run does, and
asking for 700 GB everywhere means they queue behind each other on the few
large nodes.

    python fan_bench.py                 # every finished run, final checkpoint
    python fan_bench.py --curves        # also every saved snapshot
    python fan_bench.py --dry-run
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path("/cephfs/mbandhu/qulip")
JOBS = ROOT / "jobs_bench"

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=b-{name}
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}G
#SBATCH --time=03:00:00
#SBATCH --output={root}/logs/%x-%j.out

cd {root}
source .venv/bin/activate
export QULIP_DEVICE=cpu QULIP_SKIP_METRICS=1 OMP_NUM_THREADS=4
mkdir -p results

echo "########## {run} {tag}"
python -u -m scripts.benchmark -cfg configs/{cfg}.yaml -cp "{ckpt}" \\
    2>&1 | tee results/bench_{run}_{tag}.txt
"""


def memory_for(cfg: str) -> int:
    """Bond dimension is 2^L, so deeper runs need more room for the test pass."""
    m = re.search(r"l(\d+)", cfg)
    layers = int(m.group(1)) if m else 2
    return {2: 200, 3: 350, 4: 500}.get(layers, 700)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", action="store_true",
                    help="also benchmark every params_ep*.pt snapshot")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    JOBS.mkdir(exist_ok=True)

    jobs = []
    for d in sorted((ROOT / "checkpoints").glob("mscoco-vqcfull-*")):
        run = d.name.replace("mscoco-vqcfull-", "")
        cfg = f"vqcfull_{run}"
        if not (ROOT / "configs" / f"{cfg}.yaml").exists():
            # A few runs were named before the configs settled.
            alt = {"fast": "vqcfull_fast", "fastn2": "vqcfull_fastn2",
                   "n3ring": "vqcfull_n3ring", "n4ring": "vqcfull_n4ring"}
            cfg = alt.get(run)
            if not cfg or not (ROOT / "configs" / f"{cfg}.yaml").exists():
                print(f"  no config for {run}, skipping")
                continue

        latest = sorted(d.glob("*/*/"), key=lambda p: p.stat().st_mtime)
        if not latest:
            continue
        inner = latest[-1]

        targets = [("final", inner / "last.pt")]
        if args.curves:
            targets += [(p.stem.replace("params_", ""), p)
                        for p in sorted(inner.glob("params_ep*.pt"))]

        for tag, ckpt in targets:
            if not ckpt.exists():
                continue
            name = f"{run}-{tag}"
            script = JOBS / f"{name}.sh"
            script.write_text(TEMPLATE.format(
                name=name, run=run, tag=tag, cfg=cfg, ckpt=ckpt,
                mem=memory_for(cfg), root=ROOT))
            jobs.append(script)

    print(f"  {len(jobs)} benchmark jobs")
    if args.dry_run:
        for j in jobs[:10]:
            print(f"    {j.name}")
        return

    for j in jobs:
        subprocess.run(["sbatch", str(j)], capture_output=True)
    print(f"  submitted. They run in parallel, so the set finishes in roughly")
    print(f"  the time one benchmark takes rather than {len(jobs) * 25 // 60} hours.")


if __name__ == "__main__":
    main()
