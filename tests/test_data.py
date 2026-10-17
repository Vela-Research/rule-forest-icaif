import json

import numpy as np
import pandas as pd

from rule_forest import data


def test_founder_matrix():
    a = data.load_founders()
    assert a.X.shape == (50, 8173) and int(a.y.sum()) == 158
    assert int(a.blank.sum()) == 417 and not a.X[a.blank].any()
    assert len(set(a.ids)) == len(a.ids)


def test_answer_files_hold_answers_only():
    for g in data.GENERATORS:
        df = pd.read_csv(data.FOUNDERS / "answers" / f"{g}.csv", dtype=str, keep_default_na=False)
        assert list(df.columns[:1]) == ["question"] and df.question.iloc[0] == "Success"
        assert all(len(c) == 5 for c in df.columns[1:]), "only founder columns, no summary statistics"


def test_expert_questions_are_appended():
    a, e = data.load_founders(), data.load_founders(expert=True)
    assert e.X.shape == (58, 8173) and list(e.source).count("expert") == 8
    assert np.array_equal(e.X[:50], a.X) and np.array_equal(e.y, a.y)


def test_other_generators():
    for g, q in (("claude_sonnet_4_5", 22), ("gpt_5_2", 21)):
        a = data.load_founders(g)
        assert a.X.shape == (q, 8171) and int(a.y.sum()) == 157


def test_reasked_blanks_fill_blanks_only():
    a = data.load_founders()
    X = data.load_reasked_blanks(a)
    changed = X != a.X
    assert changed.sum() == 102 and a.blank[changed].all()


def test_kiva_without_generation_loans():
    k = data.load_kiva()
    assert k.X.shape == (14, 4960) and int(k.y.sum()) == 202
    full = data.load_kiva(drop_generation_loans=False)
    assert full.X.shape == (14, 5000) and int(full.y.sum()) == 222


def test_kiva_generation_loans_match_the_questions():
    """Question 6 excludes 11 named countries; the LLM must have seen funded loans from each."""
    gen = json.loads((data.KIVA / "generation_loans.json").read_text())
    loans = pd.read_csv(data.KIVA / "loans.csv").set_index("id")
    named = json.loads((data.KIVA / "question_facts.json").read_text())[5]["value"]
    funded = loans.loc[[g["id"] for g in gen if g["label"] == 0], "country"]
    assert len(gen) == 40 and sum(g["label"] for g in gen) == 20
    assert set(named) <= set(funded)


def test_kiva_realignment():
    raw = data.load_kiva(drop_generation_loans=False, realign=False)
    fixed = data.load_kiva(drop_generation_loans=False)
    facts = data.kiva_facts(raw.ids)
    moved = (raw.X != fixed.X).any(0) | fixed.blank.any(0)
    assert int(fixed.blank.sum()) == 1433 and (fixed.blank.sum(0) <= 1).all()
    assert (fixed.X[:, ~moved] == raw.X[:, ~moved]).all()
    agree_before = (raw.X[:, moved] == facts[:, moved]).mean()
    agree_after = (fixed.X[:, moved] == facts[:, moved])[~fixed.blank[:, moved]].mean()
    assert agree_before < 0.9 < 0.99 < agree_after


def test_prompting_alignment():
    a = data.load_founders()
    p = data.load_prompting("few_shot_o3", a.ids)
    assert len(p) == 8173 and int(p.sum()) == 402 and int((p & a.y).sum()) == 31
