"""Engineered-ML features for the founder dataset, parsed deterministically from the
anonymised founder prose summaries. No LLM, no network, no external data -- every feature
is a regex/keyword expression on `summary`, so the matrix is fully reproducible from the
shipped `03_full_cross_validation_test_data_anonymised.csv`.

This re-applies the *engineered-features method* (hand-crafted structured features fed to an
elastic-net) to this dataset; it does NOT reuse any feature set developed on other data.
~30 features across education, seniority/roles, experience, company context, prior exits,
public profile, founding-team, and text statistics.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd


def _has(patterns):
    rx = re.compile("|".join(patterns))
    return lambda t: int(bool(rx.search(t)))


# ---- keyword detectors (operate on lowercased summary) -------------------------------------
_F = {
    # education
    "edu_phd":            _has([r"\bph\.?d\b", r"\bdoctorate\b", r"\bdoctoral\b"]),
    "edu_mba":            _has([r"\bmba\b", r"master of business"]),
    "edu_masters":        _has([r"\bm\.?s\b", r"\bm\.?a\b", r"master'?s", r"master of"]),
    "edu_bachelor":       _has([r"\bb\.?s\b", r"\bb\.?a\b", r"bachelor"]),
    "edu_prestige":       _has([r"prestigious", r"top business school", r"top-tier", r"elite",
                                r"renowned", r"ivy", r"highly ranked", r"leading university"]),
    "edu_stem":           _has([r"engineering", r"computer science", r"\bphysics\b", r"chemistry",
                                r"biology", r"mathematic", r"electrical", r"materials", r"\bstem\b",
                                r"data science", r"machine learning"]),
    "edu_business_field": _has([r"business", r"\bfinance\b", r"economics", r"\bmarketing\b", r"\bmba\b"]),
    # roles / seniority
    "role_ceo":           _has([r"\bceo\b", r"chief executive"]),
    "role_cto":           _has([r"\bcto\b", r"chief technolog"]),
    "role_cxo_other":     _has([r"\bcoo\b", r"\bcfo\b", r"\bcmo\b", r"\bcio\b", r"c-level",
                                r"c-suite", r"chief \w+ officer"]),
    "role_founder":       _has([r"\bfounder\b", r"\bfounded\b", r"co-found"]),
    "role_president_vp":  _has([r"president", r"vice president", r"\bvp\b", r"\bsvp\b", r"\bevp\b"]),
    "role_director":      _has([r"\bdirector\b", r"managing director", r"founding director"]),
    "role_manager":       _has([r"\bmanager\b", r"managing partner", r"management"]),
    "role_professor":     _has([r"professor", r"faculty", r"researcher at", r"national laborator"]),
    "role_leadership":    _has([r"leadership", r"\bled\b", r"\bleads\b", r"heading", r"headed"]),
    # experience depth
    "exp_extensive":      _has([r"extensive experience", r"significant experience", r"deep experience",
                                r"strong background", r"seasoned", r"veteran", r"decades"]),
    "exp_serial":         _has([r"previous startups", r"multiple companies", r"serial",
                                r"several (?:companies|ventures|startups)", r"prior ventures"]),
    # prior outcomes (handle the common negated phrasing "none ... acquired")
    "exit_ipo":           _has([r"\bipo\b", r"public offering", r"went public", r"nasdaq", r"nyse"]),
    # company context
    "co_major":           _has([r"major (?:media|technology|research|company|corporation)",
                                r"large company", r"multinational", r"fortune", r"industry leader",
                                r"public (?:tech|company)"]),
    "co_small":           _has([r"small (?:firm|company|team|startup|business)", r"early-stage"]),
    "co_public":          _has([r"public (?:company|tech)", r"publicly traded", r"publicly-traded"]),
    # public profile
    "prof_press":         _has([r"\bpress\b", r"media coverage", r"featured in", r"press coverage"]),
    "prof_awards":        _has([r"\bawards?\b", r"\bhonors?\b", r"recognition", r"\bprize\b",
                                r"accolade"]),
    "prof_strong_brand":  _has([r"strong personal brand", r"prominent", r"well-known", r"influential",
                                r"significant public presence", r"notable public"]),
    "prof_limited_brand": _has([r"limited personal brand", r"limited public presence",
                                r"little public", r"minimal (?:press|public)"]),
    # founding team
    "team_solo":          _has([r"\balone\b", r"by themselves", r"\bsolo\b", r"sole founder",
                                r"individually", r"on their own"]),
    "team_cofounders":    _has([r"co-founders", r"team of co", r"with a (?:small|large|moderate)? ?team",
                                r"founding team"]),
}


def build_features(summaries: pd.Series) -> pd.DataFrame:
    txt = summaries.astype(str).str.lower()
    feats = {name: txt.map(fn).astype(int) for name, fn in _F.items()}

    # prior-acquisition with negation handling
    acquired = txt.str.contains(r"acqui", regex=True)
    neg_acq = txt.str.contains(r"(?:none|not|no|never|haven'?t|hasn'?t).{0,40}acqui", regex=True) | \
              txt.str.contains(r"acqui\w*\.?\s*(?:none|not|no)", regex=True)
    feats["exit_acquired_pos"] = (acquired & ~neg_acq).astype(int)

    # ordinal: "worked full-time at a {small|moderate|large} number of companies"
    def n_comp(t):
        if re.search(r"(?:large|many|numerous|significant) number of comp", t): return 3
        if re.search(r"moderate number of comp", t): return 2
        if re.search(r"small number of comp", t): return 1
        return 0
    feats["n_companies_ord"] = txt.map(n_comp).astype(int)

    # education level ordinal (max degree)
    def edu_ord(t):
        if re.search(r"\bph\.?d\b|doctorate|doctoral", t): return 4
        if re.search(r"\bmba\b|master", t): return 3
        if re.search(r"bachelor|\bb\.?s\b|\bb\.?a\b", t): return 2
        return 0
    feats["edu_level_ord"] = txt.map(edu_ord).astype(int)

    # text statistics
    feats["txt_words"] = txt.str.split().map(len).astype(int)
    feats["txt_roles_mentions"] = txt.str.count(r"\brole|\bserved as|\bworked as|\bheld\b").astype(int)
    feats["txt_industry_mentions"] = txt.str.count(r"industr").astype(int)

    return pd.DataFrame(feats)


if __name__ == "__main__":
    REL = Path(__file__).resolve().parents[3]
    d = pd.read_csv(REL / "rrf_submission" / "precomputed" /
                    "03_full_cross_validation_test_data_anonymised.csv")
    X = build_features(d["summary"])
    y = d["startup_success"].astype(int)
    print(f"features: {X.shape[1]}  founders: {X.shape[0]}  positives: {int(y.sum())}")
    # positive-rate lift per binary feature (quick signal check)
    base = y.mean()
    print(f"\nbase rate = {base:.4f}. Per-feature P(success | feature=1) / lift:")
    for c in X.columns:
        col = X[c]
        if set(col.unique()) <= {0, 1} and col.sum() > 20:
            p1 = y[col == 1].mean()
            print(f"  {c:22s} n1={int(col.sum()):5d}  P(succ|1)={p1:.4f}  lift={p1/base:5.2f}x")
