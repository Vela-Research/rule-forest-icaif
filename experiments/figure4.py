"""Figure 4. When the equal-weight vote beats learned weights, on evaluation founders only.

Each point subsamples successes and failures from the 8,173 scored evaluation founders. The vote
and the elastic net (sweep configuration) are then fitted under the same 10 x 10 nested
cross-validation. In every outer training fold the candidate pool is rebuilt from the 50
GPT-4o-mini questions using training answers and labels only, and both models use that pool.
The precision floor is scaled to the training fold's base rate (0.02001 at the real 1.93%), since
a fixed floor would drop every question at 0.5% and none at 9%.

Successes = sample size x base rate, so each slice holds one of them fixed:
  S4  base rate 1.93%           successes 15, 30, 60, 100, 158      panels (a), (c)
  S1  base rate 5%              successes 15, 30, 60, 100, 158      panels (b), (c)
  S5  base rate 9%              successes 15, 30, 60, 100, 158      panel (c)
  S2  sample size 3,160         successes 15, 30, 60, 100, 158      panels (c), (e)
  S3  158 successes             base rate 1.93%, 3%, 5%, 9%         panel (d)
Ten seeded draws per point; the full 8,173-founder point is a single draw.

  python experiments/figure4.py run [--jobs 16]   # about 5 CPU-hours -> results/figure4_draws.jsonl
  python experiments/figure4.py plot               # -> results/figure4.json, figures/figure4.pdf
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
from scipy.stats import spearmanr

from experiments import sweeps
from rule_forest import data, vote
from rule_forest.pool import PRECISION_FLOOR, candidate_pool

DRAWS = ROOT / "results" / "figure4_draws.jsonl"
SUCCESSES = [15, 30, 60, 100, 158]
SEED_LABEL = "a07"          # label hashed into every subsample seed; fixes the draws
N_DRAWS = 10
TEAL, NAVY, GREY = "#1f7a8c", "#022b3a", "#6b6a63"
RAMP = {"S4": "#8fcbd6", "S1": "#1f7a8c", "S5": "#0b3c47"}
LABEL = {"S4": "1.93% fixed", "S1": "5% fixed", "S5": "9% fixed"}


def points(n_pos_all=158, n_neg_all=8015):
    """{(successes, failures): [slices]} for every point."""
    real = n_pos_all / (n_pos_all + n_neg_all)
    pts = {}
    def add(sl, p, n):
        pts.setdefault((p, min(n, n_neg_all)), []).append(sl)
    for p in SUCCESSES:
        add("S1", p, round(p * 0.95 / 0.05))
        add("S2", p, 3160 - p)
        add("S4", p, round(p * n_neg_all / n_pos_all))
        add("S5", p, round(p * 0.91 / 0.09))
    for prev in (real, 0.03, 0.05, 0.09):
        add("S3", 158, round(158 * (1 - prev) / prev))
    return pts


_A = None


def _init():
    global _A
    _A = data.load_founders("gpt_4o_mini")


def _work(task):
    n_pos, n_neg, d = task["key"]
    idx = sweeps.subsample(_A.y, n_pos, n_neg, sweeps.draw_seed(SEED_LABEL, n_pos, n_neg, d))
    X, y = _A.X[:, idx], _A.y[idx]
    real = _A.y.mean()
    pools = {}
    for rep, k, tr, _ in vote.outer_folds(y):
        pools[(rep, k)] = candidate_pool(X, y, tr, floor=PRECISION_FLOOR * y[tr].mean() / real)
    return {"key": task["key"], "slices": task["slices"], "n_pos": n_pos, "n_neg": n_neg,
            "prevalence": n_pos / (n_pos + n_neg), **sweeps.compare(X, y, pools)}


def run(jobs, fresh=False):
    tasks = []
    for (p, n), sl in sorted(points().items()):
        full = p >= 158 and n >= 8015
        tasks += [{"key": [p, n, d], "slices": sl} for d in range(1 if full else N_DRAWS)]
    sweeps.run(tasks, _work, _init, DRAWS, jobs, fresh)


def summarise(rows):
    """Per slice, the mean over draws at each point, the draws the vote leads, and Spearman's
    rho of the per-draw gap against successes (or base rate for S3), as a descriptive trend."""
    out = {}
    for sl in ("S1", "S2", "S3", "S4", "S5"):
        rs = [r for r in rows if sl in r["slices"]]
        table = []
        for key in sorted({(r["n_pos"], r["n_neg"]) for r in rs}):
            g = [r for r in rs if (r["n_pos"], r["n_neg"]) == key]
            row = {"successes": key[0], "n": sum(key), "base_rate": key[0] / sum(key), "draws": len(g),
                   "vote_leads_in_draws": int(sum(r["gap"] > 0 for r in g))}
            for f in ("vote_f05", "en_f05", "gap", "vote_optimism", "en_optimism",
                      "vote_selection_jaccard", "en_support_jaccard"):
                row[f] = float(np.mean([r[f] for r in g]))
            table.append(row)
        x = [r["prevalence"] if sl == "S3" else r["n_pos"] for r in rs]
        out[sl] = {"table": table, "spearman_gap": float(spearmanr(x, [r["gap"] for r in rs]).statistic)}
    pts = {}
    for r in rows:
        pts.setdefault((r["n_pos"], r["n_neg"]), []).append(r)
    larger = sum(np.mean([r["en_optimism"] for r in g]) > np.mean([r["vote_optimism"] for r in g])
                 for g in pts.values())
    return {"slices": out, "points": len(pts), "draws": len(rows),
            "elastic_net_overfits_more_at_points": int(larger),
            "note": "optimism = mean training-fold F0.5 minus held-out pooled F0.5, averaged over draws"}


def plot(summary, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = summary["slices"]
    t = lambda sl: S[sl]["table"]

    def lab(ax, s, dx=-42):
        ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 7), textcoords="offset points",
                    fontsize=13, fontweight="bold", va="bottom", ha="left", annotation_clip=False)

    def pair(ax, rows):
        x = [r["successes"] for r in rows]
        ax.plot(x, [r["vote_f05"] for r in rows], "s-", color=TEAL, lw=2, ms=6, label="equal-weight vote")
        ax.plot(x, [r["en_f05"] for r in rows], "o--", color=NAVY, lw=2, ms=5, label="elastic net")

    fig = plt.figure(figsize=(12, 5.8))
    gt = fig.add_gridspec(1, 2, left=0.07, right=0.985, top=0.92, bottom=0.62, wspace=0.30)
    gb = fig.add_gridspec(1, 2, left=0.07, right=0.985, top=0.40, bottom=0.11, width_ratios=[2.1, 1], wspace=0.22)
    gcd = gb[0, 0].subgridspec(1, 2, width_ratios=[1.35, 1], wspace=0.10)

    ax = fig.add_subplot(gt[0, 0]); pair(ax, t("S4"))
    ax.set_title("Real base rate (1.93%), evaluation founders", fontsize=11)
    ax.set_xlabel("Successes in the sample"); ax.set_ylabel("Held-out $F_{0.5}$")
    ax.set_ylim(0, 0.10); ax.legend(fontsize=8.5, loc="lower right"); lab(ax, "(a)")

    ax = fig.add_subplot(gt[0, 1]); pair(ax, t("S1"))
    ax.set_title("Base rate fixed at 5%, evaluation founders", fontsize=11)
    ax.set_xlabel("Successes in the sample"); ax.set_ylabel("Held-out $F_{0.5}$")
    ax.set_ylim(0, 0.20); ax.legend(fontsize=8.5, loc="lower right"); lab(ax, "(b)")

    axc = fig.add_subplot(gcd[0, 0])
    for sl in ("S4", "S1", "S5"):
        axc.plot([r["successes"] for r in t(sl)], [r["gap"] for r in t(sl)], "s-", color=RAMP[sl], lw=1.8, ms=4,
                 label=LABEL[sl])
    axc.plot([r["successes"] for r in t("S2")], [r["gap"] for r in t("S2")], "D:", color=GREY, lw=1.8, ms=4,
             label="n fixed (3,160)")
    axc.axhline(0, color="grey", lw=0.8, ls=":")
    axc.set_title("Base rate or sample size held fixed", fontsize=9)
    axc.set_xlabel("Successes in the sample", fontsize=8)
    axc.set_ylabel("Vote minus elastic net\n(held-out $F_{0.5}$)", fontsize=8)
    axc.tick_params(labelsize=7); axc.legend(fontsize=7, loc="upper right", frameon=False); lab(axc, "(c)")

    axd = fig.add_subplot(gcd[0, 1], sharey=axc)
    xs = [100 * r["base_rate"] for r in t("S3")]
    axd.plot(xs, [r["gap"] for r in t("S3")], "s-", color=TEAL, lw=1.8, ms=5)
    axd.axhline(0, color="grey", lw=0.8, ls=":")
    axd.set_xticks(xs); axd.set_xticklabels([f"{v:.2f}" if v < 2 else f"{v:.0f}" for v in xs])
    axd.tick_params(labelsize=7, labelleft=False)
    axd.set_title("158 successes, base rate raised", fontsize=9)
    axd.set_xlabel("Base rate (%)", fontsize=8); lab(axd, "(d)", dx=-20)

    axe = fig.add_subplot(gb[0, 1])
    x = [r["successes"] for r in t("S2")]
    axe.plot(x, [r["vote_selection_jaccard"] for r in t("S2")], "s-", color=TEAL, lw=1.8, ms=5,
             label="vote: questions it selects")
    axe.plot(x, [r["en_support_jaccard"] for r in t("S2")], "o--", color=NAVY, lw=1.8, ms=5,
             label="elastic net: non-zero weights")
    axe.set_title("Same questions chosen across folds (n = 3,160)", fontsize=9)
    axe.set_xlabel("Successes in the sample", fontsize=8)
    axe.set_ylabel("Share of chosen questions\ntwo folds have in common", fontsize=8)
    axe.tick_params(labelsize=7); axe.set_ylim(0, 1); axe.legend(fontsize=7, frameon=False, loc="lower right")
    lab(axe, "(e)")

    out_pdf.parent.mkdir(exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["run", "plot"])
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--fresh", action="store_true", help="recompute every draw instead of resuming")
    a = ap.parse_args()
    if a.step == "run":
        run(a.jobs, a.fresh)
        return
    summary = summarise(sweeps.read(DRAWS))
    (ROOT / "results" / "figure4.json").write_text(json.dumps(summary, indent=1) + "\n")
    plot(summary, ROOT / "figures" / "figure4.pdf")
    for sl, s in summary["slices"].items():
        print(sl, f"rho {s['spearman_gap']:+.2f}",
              " ".join(f"{r['successes']}:{r['gap']:+.3f}({r['vote_leads_in_draws']}/{r['draws']})" for r in s["table"]))
    print(f"elastic net overfits more at {summary['elastic_net_overfits_more_at_points']} of {summary['points']} points")


if __name__ == "__main__":
    main()
