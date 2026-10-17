"""Rebuild every result and figure in the paper from data/.

  python reproduce.py            # recompute everything (about 25 CPU-hours; about 2 hours on 16 cores)
  python reproduce.py --quick    # reuse the committed model predictions and sweep draws (minutes)

The slow steps are the cross-validated model fits. --quick skips them and recomputes every number
from the predictions and draws already in results/. Afterwards, `git status results figures`
shows whether anything differs from the committed version, and `python -m pytest` checks the
numbers the paper reports.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(args, slow=False, quick=False, needs=None):
    if slow and quick:
        print(f"skip (--quick): {' '.join(args)}", flush=True)
        return
    if needs and subprocess.run([sys.executable, "-c", f"import {needs}"], capture_output=True).returncode:
        print(f"skip ({needs} cannot be imported; the committed result is kept): {' '.join(args)}", flush=True)
        return
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the slow model fits")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    q, j = a.quick, ["--jobs", str(a.jobs)]
    step(["baselines/gptree_rrm/rescore.py"])
    step(["baselines/engineered_ml/run.py", "--nproc", str(a.jobs)], slow=True, quick=q)
    step(["baselines/embeddings/run.py", "--nproc", str(a.jobs)], slow=True, quick=q, needs="xgboost")
    step(["experiments/questions.py"])
    step(["experiments/founders.py", "predict", *j], slow=True, quick=q)
    step(["experiments/founders.py", "report"])
    step(["experiments/figure4.py", "run", "--fresh", *j], slow=True, quick=q)
    step(["experiments/figure4.py", "plot"])
    step(["experiments/figure5.py", "run", "--fresh", *j], slow=True, quick=q)
    step(["experiments/figure5.py", "plot"])
    step(["experiments/figure5.py", "run", "--fresh", "--as-scored", *j], slow=True, quick=q)
    step(["experiments/figure5.py", "plot", "--as-scored"])
    step(["experiments/paper_numbers.py"])


if __name__ == "__main__":
    main()
