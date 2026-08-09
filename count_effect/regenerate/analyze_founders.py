"""Regenerate count_effect/data/cache/summary_founders.json from results_founders.jsonl.

Computes the founder natural-prevalence count-effect statistics the paper cites:
  natural  : full-pool Delta = F0.5(unit-vote) - F0.5(elastic-net) at natural prevalence.
  npos_nat : vary positive COUNT at natural prevalence -> Spearman rho(n_pos, Delta),
             draw-level permutation p for H1 (rho<0, B=10000), and sign-change crossover.

Statistics ported verbatim from the experiment-side analyze_crossdomain (draw-level Spearman;
permutation RNG seed 20260730). REPORT-ONLY; reads only results_founders.jsonl.
  python analyze_founders.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data" / "cache"
SRC = CACHE / "results_founders.jsonl"
OUT = CACHE / "summary_founders.json"

B_PERM = 10000
RNG = np.random.RandomState(20260730)


def perm_p(xn, dl, rho_obs, B=B_PERM):
    """One-sided draw-level permutation p for H1 (rho < 0): shuffle Delta across draws."""
    xn = np.asarray(xn, float); dl = np.asarray(dl, float)
    if len(xn) < 4 or np.allclose(dl, dl[0]):
        return float("nan")
    cnt = 0
    for _ in range(B):
        rp = spearmanr(xn, RNG.permutation(dl)).statistic
        if rp <= rho_obs + 1e-12:
            cnt += 1
    return (1 + cnt) / (B + 1)


def crossover(pts):
    for a, b in zip(pts, pts[1:]):
        ga, gb = a["gap_mean"], b["gap_mean"]
        if ga == 0:
            return float(a["x"])
        if ga > 0 >= gb or ga < 0 <= gb:
            if gb != ga:
                return float(a["x"] + (b["x"] - a["x"]) * (0 - ga) / (gb - ga))
            return float(a["x"])
    return None


def grid_means(arm_rows, xkey="n_pos"):
    by_pt = {}
    for r in arm_rows:
        by_pt.setdefault(r["point_id"], []).append(r)
    pts = []
    for pid, rs in by_pt.items():
        g = np.array([r["gap"] for r in rs], float)
        x = float(np.mean([r[xkey] for r in rs]))
        sd = float(g.std(ddof=1)) if len(g) > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(len(g)) if len(g) > 1 else 0.0
        pts.append({"point_id": pid, "x": x, "n_draws": len(rs),
                    "gap_mean": float(g.mean()), "gap_sd": sd, "ci95": float(ci),
                    "prev_mean": float(np.mean([r["prevalence"] for r in rs]))})
    pts.sort(key=lambda d: d["x"])
    return pts


def analyze_arm(arm_rows, xkey="n_pos"):
    xn = [r[xkey] for r in arm_rows]
    dl = [r["gap"] for r in arm_rows]
    rho = spearmanr(xn, dl).statistic if len(set(xn)) > 1 else float("nan")
    p = perm_p(xn, dl, rho) if not np.isnan(rho) else float("nan")
    pts = grid_means(arm_rows, xkey)
    return {"rho": float(rho), "perm_p": (None if np.isnan(p) else float(p)),
            "n_draws_total": len(arm_rows), "n_points": len(pts),
            "outer_cv_repeats": sorted({r.get("n_repeats") for r in arm_rows}),
            "outer_cv_splits": sorted({r.get("n_splits") for r in arm_rows}),
            "inner_selection_folds": sorted({r.get("inner_splits") for r in arm_rows}),
            "crossover": crossover(pts), "points": pts}


def main():
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    nat = [r for r in rows if r["arm"] == "natural"]
    sweep = [r for r in rows if r["arm"] == "npos_nat"]
    base_rate = rows[0]["base_rate"] if rows else None
    res = analyze_arm(sweep, "n_pos") if sweep else None
    natural = None
    if nat:
        nr = nat[0]
        natural = {"n": nr["n"], "n_pos": nr["n_pos"], "prevalence": nr["prevalence"],
                   "delta_natural": nr["gap"]}
    summary = {
        "dataset": "founders", "display": "Founders", "base_rate": base_rate,
        "protocol": {"outer_cv": "10x10 repeated nested", "inner_selection_folds": 5,
                     "metric": "F0.5", "draws_per_point": 10, "perm_B": B_PERM,
                     "gap": "Delta = F0.5(unit-vote) - F0.5(elastic-net)",
                     "npos_nat": "vary positive COUNT at natural prevalence"},
        "natural": natural, "npos_nat": res,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT.name}")
    if natural:
        print(f"  base_rate={base_rate*100:.2f}%  natural Delta={natural['delta_natural']:+.4f}")
    if res:
        print(f"  rho(n_pos, gap) = {res['rho']:+.3f}   perm_p = {res['perm_p']}   "
              f"crossover = {res['crossover']}")
        for p in res["points"]:
            print(f"    n_pos={p['x']:6.1f} gap={p['gap_mean']:+.4f} +/-{p['ci95']:.4f} (nd={p['n_draws']})")


if __name__ == "__main__":
    main()
