"""How the founder questions were produced, and the missing-answer policy (Sections 4.1 and 4.2).

1. Generation. The 497 question-generation founders are split into prompts of up to 20 founders,
   half successful where possible (split_into_sets below, as the generation pipeline ran it).
   Each prompt shows only the founders' profiles, never their labels. The LLM writes YES/NO
   questions for each prompt.
2. Chain of question counts: generated, exact-unique, after semantic de-duplication, passing the
   validation screen (precision >= 0.10 and recall >= 0.10 on the validation founders), and
   scored on the evaluation founders.
3. Blank answers. Every number in the paper counts a blank answer as NO. Some blanks were later
   re-asked with the same prompt; results/predictions/vote_hamming_0.15_reasked_blanks.npz holds
   the vote with those answers filled in.

  python experiments/questions.py   # -> results/questions.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from rule_forest import data

CHAIN = data.FOUNDERS / "question_chain"


def split_into_sets(df, founders_per_set=20):
    """The generation pipeline's sampler: sets of up to 20 founders, balanced where possible,
    each drawn with random_state 42 + set number and shuffled."""
    succ, fail = df[df.startup_success == 1].copy(), df[df.startup_success == 0].copy()
    sets, k = [], 0
    while len(succ) or len(fail):
        n_s = min(len(succ), founders_per_set // 2)
        n_f = min(len(fail), founders_per_set - n_s)
        if n_s + n_f < founders_per_set:
            if len(succ) > n_s:
                n_s += min(founders_per_set - (n_s + n_f), len(succ) - n_s)
            elif len(fail) > n_f:
                n_f += min(founders_per_set - (n_s + n_f), len(fail) - n_f)
        if n_s == 0 and n_f == 0:
            break
        s = succ.sample(n=n_s, random_state=42 + k) if n_s else succ.iloc[:0]
        f = fail.sample(n=n_f, random_state=42 + k) if n_f else fail.iloc[:0]
        succ, fail = succ.drop(s.index), fail.drop(f.index)
        sets.append(pd.concat([s, f]).sample(frac=1, random_state=42 + k).reset_index(drop=True))
        k += 1
    return sets


def generation():
    df = pd.read_csv(data.FOUNDERS / "profiles" / "generation.csv")
    sets = split_into_sets(df)
    shown = pd.Series([u for s in sets for u in s.loc[s.startup_success == 1, "founder_uuid"]]).value_counts()
    return {"founders": len(df), "successes": int(df.startup_success.sum()), "prompts": len(sets),
            "founders_per_prompt": [len(s) for s in sets],
            "successes_per_prompt": [int(s.startup_success.sum()) for s in sets],
            "distinct_successes_shown": int(len(shown)), "most_times_one_success_is_shown": int(shown.max())}


def chain():
    norm = lambda s: s.astype(str).str.strip().str.lower()
    n = lambda f: len(pd.read_csv(CHAIN / f))
    val = pd.read_csv(CHAIN / "validation_answers.csv", dtype=str, keep_default_na=False).set_index("question")
    y = (val.loc["Success"] == "1").to_numpy()
    q = val.drop(index="Success")
    X = (q.to_numpy() == "1")
    tp, fp = (X & y).sum(1), (X & ~y).sum(1)
    prec = np.divide(tp, tp + fp, out=np.zeros(len(q)), where=(tp + fp) > 0)
    rec = tp / y.sum()
    passes = (prec >= 0.10) & (rec >= 0.10)
    scored = set(norm(pd.Series(data.load_founders().questions)))
    is_scored = norm(pd.Series(q.index)).isin(scored).to_numpy()
    show = lambda m: [{"question": str(t), "precision": float(p), "recall": float(r)}
                      for t, p, r in zip(q.index[m], prec[m], rec[m])]
    return {"generated": n("generated.csv"), "exact_unique": n("generated_exact_unique.csv"),
            "after_semantic_dedup": n("after_semantic_dedup.csv"),
            "scored_on_evaluation_founders": int(is_scored.sum()),
            "validation_founders_in_file": int(len(y)), "validation_successes": int(y.sum()),
            "pass_screen_in_this_file": int(passes.sum()),
            "scored_but_failing_screen_in_this_file": show(is_scored & ~passes),
            "passing_screen_in_this_file_but_not_scored": show(passes & ~is_scored),
            "note": "The screen keeps questions with validation precision >= 0.10 and recall >= 0.10. "
                    "Recomputed on the shipped validation answers it does not match the 50 scored questions "
                    "exactly (the lists above), so the release documents the 50 that were scored."}


def blanks():
    out = {}
    for g in data.GENERATORS:
        a = data.load_founders(g)
        out[g] = {"cells": int(a.blank.size), "blank": int(a.blank.sum()), "blank_share": float(a.blank.mean())}
    a = data.load_founders()
    filled = data.load_reasked_blanks(a)
    out["gpt_4o_mini"].update({"reasked": int(len(pd.read_csv(data.FOUNDERS / "answers" / "gpt_4o_mini_reasked_blanks.csv"))),
                               "reasked_yes": int((filled != a.X).sum())})
    return out


def founders():
    ev = pd.read_csv(data.FOUNDERS / "profiles" / "evaluation.csv")
    prefix = ev.founder_uuid.str[:5]
    a = data.load_founders()
    return {"evaluation_profiles": len(ev), "evaluation_successes": int(ev.startup_success.sum()),
            "distinct_5_character_ids": int(prefix.nunique()),
            "ids_shared_by_two_founders": int((prefix.value_counts() > 1).sum()),
            "founders_in_gpt_4o_mini_matrix": int(len(a.ids)), "successes_in_matrix": int(a.y.sum()),
            "profile_rows_not_in_matrix": [int(i) for i, p in enumerate(prefix) if p not in set(a.ids)],
            "note": "Founder ids are the first 5 characters of an anonymised id, so 27 pairs of founders share "
                    "an id and appear once. The answering run sent founders in batches of 20 and the last "
                    "4 founders were never sent, so the matrix has 8,177 - 4 = 8,173 founders."}


def main():
    out = {"generation": generation(), "question_chain": chain(), "blank_answers": blanks(), "founders": founders()}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "questions.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
