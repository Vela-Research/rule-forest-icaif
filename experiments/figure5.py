"""Figure 5. The same comparison on Kiva microloans.

The 14 Kiva questions were written by an LLM shown 40 labelled training loans
(data/kiva/generation_loans.json); those loans are removed, leaving 4,960 loans of which 202
expired unfunded (4.1%). Short LLM replies are re-aligned (rule_forest.data.load_kiva). Every
funded loan is kept and the expired ones are thinned, so the base rate falls towards the founder
rate. At each count the vote (all 14 questions, ranked in each training fold, with no candidate-pool
step) and the sweep elastic net are fitted under the same 10 x 10 nested cross-validation, each
with 5 inner folds. Ten seeded draws per count; the full data is a single draw.

  python experiments/figure5.py run [--jobs 16]   # about 1 CPU-hour -> results/figure5_draws.jsonl
  python experiments/figure5.py plot               # -> results/figure5.json, figures/figure5.pdf
  --as-scored  uses the scores as returned, without re-alignment (results/figure5_as_scored*.json*)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker; set before numpy loads
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats

from experiments import sweeps
from rule_forest import data, vote

DRAWS = ROOT / "results" / "figure5_draws.jsonl"
DRAWS_AS_SCORED = ROOT / "results" / "figure5_as_scored_draws.jsonl"
COUNTS = [15, 30, 60, 93, 130, 180]      # 93 successes gives the founder base rate, 1.9%
SEED_LABEL = "kiva_D2_clean"              # label hashed into every subsample seed; fixes the draws
N_DRAWS = 10
TEAL, NAVY = "#1f7a8c", "#022b3a"

_A = None


def _init(realign=True):
    global _A
    _A = data.load_kiva(realign=realign)


def _work(task):
    count, d = task["key"]
    n_neg = int((_A.y == 0).sum())
    idx = sweeps.subsample(_A.y, count, n_neg, sweeps.draw_seed(SEED_LABEL, count, d))
    X, y = _A.X[:, idx], _A.y[idx]
    pools = {(rep, k): list(range(len(X))) for rep, k, _, _ in vote.outer_folds(y)}
    return {"key": task["key"], "count": count, "prevalence": float(y.mean()), **sweeps.compare(X, y, pools)}


def _init_as_scored():
    _init(realign=False)


def run(jobs, fresh=False, as_scored=False):
    full = int(data.load_kiva().y.sum())
    tasks = [{"key": [c, d]} for c in COUNTS for d in range(N_DRAWS)] + [{"key": [full, 0]}]
    sweeps.run(tasks, _work, _init_as_scored if as_scored else _init,
               DRAWS_AS_SCORED if as_scored else DRAWS, jobs, fresh)


def summarise(rows):
    table = []
    for c in sorted({r["count"] for r in rows}):
        g = [r for r in rows if r["count"] == c]
        row = {"successes": c, "draws": len(g), "base_rate": float(np.mean([r["prevalence"] for r in g])),
               "vote_leads_in_draws": int(sum(r["gap"] > 0 for r in g))}
        for f in ("vote_f05", "en_f05", "gap"):
            v = np.array([r[f] for r in g])
            row[f] = float(v.mean())
            row[f + "_ci95_halfwidth"] = (float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
                                          if len(v) > 1 else 0.0)
        table.append(row)
    return {"table": table, "loans": len(data.load_kiva().y), "note": "95% t-intervals over draws"}


def plot(summary, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = summary["table"]
    K = [r["successes"] for r in t]
    full = max(K)
    prev = [100 * r["base_rate"] for r in t]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.errorbar(K, [r["vote_f05"] for r in t], yerr=[r["vote_f05_ci95_halfwidth"] for r in t], fmt="s-",
                color=TEAL, lw=2, ms=6, capsize=3, label="equal-weight vote")
    ax.errorbar(K, [r["en_f05"] for r in t], yerr=[r["en_f05_ci95_halfwidth"] for r in t], fmt="o--",
                color=NAVY, lw=2, ms=5, capsize=3, label="elastic net")
    ax.axvline(93, color="grey", ls=":", lw=1.2)
    ax.text(90, 0.208, "1.9%  (founder rate)", rotation=90, va="top", ha="right", fontsize=8, color="grey")
    ax.axvline(full, color="grey", ls=":", lw=1.2)
    ax.text(full - 3, 0.004, f"full data ({prev[-1]:.1f}%)", rotation=90, va="bottom", ha="right", fontsize=8,
            color="grey")
    ax.set_xlabel("Number of positive examples"); ax.set_ylabel("Held-out $F_{0.5}$")
    ax.set_xlim(0, full + 8); ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_ylim(0, 0.22); ax.set_yticks([0, 0.1, 0.2]); ax.legend(fontsize=9, loc="upper left")
    axt = ax.secondary_xaxis("top"); axt.set_xticks(K); axt.set_xticklabels([f"{p:.1f}" for p in prev], fontsize=7.5)
    axt.set_xlabel("base rate (%)", fontsize=9)
    fig.tight_layout()
    out_pdf.parent.mkdir(exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["run", "plot"])
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--fresh", action="store_true", help="recompute every draw instead of resuming")
    ap.add_argument("--as-scored", action="store_true", help="skip the re-alignment of short replies")
    a = ap.parse_args()
    if a.step == "run":
        run(a.jobs, a.fresh, a.as_scored)
        return
    summary = summarise(sweeps.read(DRAWS_AS_SCORED if a.as_scored else DRAWS))
    name = "figure5_as_scored" if a.as_scored else "figure5"
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(summary, indent=1) + "\n")
    if not a.as_scored:
        plot(summary, ROOT / "figures" / "figure5.pdf")
    for r in summary["table"]:
        print(f"{r['successes']:3d} ({100 * r['base_rate']:.1f}%) vote {r['vote_f05']:.3f} en {r['en_f05']:.3f} "
              f"gap {r['gap']:+.3f} vote leads {r['vote_leads_in_draws']}/{r['draws']}")


if __name__ == "__main__":
    main()
