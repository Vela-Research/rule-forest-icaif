"""Loaders for the shipped LLM answers.

Founders. Each file in data/founders/answers/ has one row per question and one column per
evaluation founder (5-character anonymised id). The first row, "Success", holds the labels.
Cells are 1 (YES), 0 (NO) or blank (the LLM reply failed or could not be read). A blank counts
as NO, the policy every number in the paper uses. Column order is the cross-validation order,
and row order is the order the questions were scored, so neither may be changed.

Kiva. data/kiva/answers.jsonl has one record per loan with a score in [0, 1] for each of the 14
questions in data/kiva/questions.json. A score of 0.5 or more counts as YES. The scoring script
padded a 13-answer reply with its own mean, which leaves every answer after the skipped question
one slot early; load_kiva puts those answers back in place (realign_short_replies).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FOUNDERS = DATA / "founders"
KIVA = DATA / "kiva"
GENERATORS = ("gpt_4o_mini", "claude_sonnet_4_5", "gpt_5_2")
KIVA_YES = 0.5


@dataclass
class Answers:
    """Binary answers of one question set. Rows are questions and columns are examples."""
    X: np.ndarray          # (Q, F) int8, 1 = YES
    y: np.ndarray          # (F,) int8, 1 = success (founders) or expired unfunded (Kiva)
    questions: list        # question text, row order
    source: np.ndarray     # (Q,) "llm" or "expert"
    ids: np.ndarray        # (F,) example ids, column order
    blank: np.ndarray      # (Q, F) bool, True where the reply was missing

    def subset(self, rows=None, cols=None) -> "Answers":
        r = slice(None) if rows is None else np.asarray(rows)
        c = slice(None) if cols is None else np.asarray(cols)
        return Answers(self.X[r][:, c], self.y[c], list(np.asarray(self.questions, dtype=object)[r]),
                       self.source[r], self.ids[c], self.blank[r][:, c])


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("question")


def _binary(frame: pd.DataFrame):
    vals = frame.to_numpy()
    if not np.isin(vals, ["0", "1", ""]).all():
        raise ValueError("answer cells must be 0, 1 or blank")
    return (vals == "1").astype(np.int8), vals == ""


def load_founders(generator: str = "gpt_4o_mini", expert: bool = False) -> Answers:
    """Evaluation-founder answers for one question generator, optionally with the 8 expert
    questions appended (GPT-4o-mini only). Expert answers missing for a founder count as NO."""
    if generator not in GENERATORS:
        raise ValueError(f"generator must be one of {GENERATORS}")
    df = _read(FOUNDERS / "answers" / f"{generator}.csv")
    labels = df.loc["Success"]
    if (labels == "").any():
        raise ValueError("blank label")
    q = df.drop(index="Success")
    source = ["llm"] * len(q)
    if expert:
        if generator != "gpt_4o_mini":
            raise ValueError("expert questions were answered for GPT-4o-mini founders only")
        e = _read(FOUNDERS / "answers" / "expert.csv").reindex(columns=q.columns, fill_value="")
        q = pd.concat([q, e])
        source += ["expert"] * len(e)
    if q.index.str.strip().str.lower().duplicated().any():
        raise ValueError("duplicate question text")
    X, blank = _binary(q)
    return Answers(X=X, y=(labels.to_numpy() == "1").astype(np.int8), questions=list(q.index),
                   source=np.array(source), ids=np.array(q.columns, dtype=str), blank=blank)


def load_reasked_blanks(answers: Answers) -> np.ndarray:
    """answers.X with the blank cells that were later re-asked (same prompt, GPT-4o-mini,
    temperature 0) filled in. 289 of the 417 blank cells were re-asked."""
    r = pd.read_csv(FOUNDERS / "answers" / "gpt_4o_mini_reasked_blanks.csv", dtype=str)
    row = {q: i for i, q in enumerate(answers.questions)}
    col = {h: j for j, h in enumerate(answers.ids)}
    X = answers.X.copy()
    for q, h, a in r.itertuples(index=False):
        i, j = row[q], col[h]
        if not answers.blank[i, j]:
            raise ValueError("re-asked cell was not blank")
        X[i, j] = int(a)
    return X


def load_prompting(name: str, ids) -> np.ndarray:
    """Direct-prompting predictions (1 = predicted successful) aligned to `ids`. A founder the
    run did not score, or scored with a reply that could not be read, counts as NO."""
    df = _read(FOUNDERS / "prompting" / f"{name}.csv")
    pred = df.loc["Prediction"]
    return np.array([1 if pred.get(h, "") == "1" else 0 for h in ids], dtype=np.int8)


PROMPTING = ("few_shot_o3", "zero_shot_o3", "few_shot_o3_mini", "zero_shot_o3_mini",
             "few_shot_gpt_4o_mini", "zero_shot_gpt_4o_mini")


def kiva_facts(ids):
    """(14, n) 0/1 truth of each Kiva question for loans `ids`, from the loan's amount, sector and
    country (data/kiva/loans.csv and question_facts.json). Every Kiva question is such a fact."""
    loans = pd.read_csv(KIVA / "loans.csv").set_index("id").loc[list(ids)]
    out = []
    for f in json.loads((KIVA / "question_facts.json").read_text()):
        col, op, v = loans[f["field"]], f["op"], f["value"]
        if op == ">":
            hit = col > v
        elif op == "<":
            hit = col < v
        elif op == "==":
            hit = col == v
        elif op == "not in":
            hit = ~col.isin(v)
        else:
            raise ValueError(op)
        out.append(hit.to_numpy())
    return np.array(out, dtype=np.int8)


def realign_short_replies(scores, facts):
    """Put back in place the answers of replies the scoring script padded.

    When the LLM returned 13 scores for the 14 questions, the script appended their mean, so the
    answers after the skipped question sit one slot early. Such rows are recognised by a last score
    equal to the mean of the first 13, strictly between 0 and 1. The skipped position is taken to be
    the one whose re-aligned answers agree best with the loan's facts (the earliest on a tie), and
    the skipped question is left blank (NO). Returns the re-aligned scores (NaN = blank) and the
    indices of the re-aligned rows. Uses no labels."""
    out = scores.astype(float).copy()
    fixed = []
    for i, v in enumerate(scores):
        m = v[:13].mean()
        if not (v[13] == v[:13].sum() / 13 and 0 < m < 1):
            continue
        yes = (v[:13] >= KIVA_YES).astype(int)
        agree = [(np.delete(facts[:, i], j) == yes).sum() for j in range(14)]
        j = int(np.argmax(agree))
        out[i] = np.insert(v[:13], j, np.nan)
        fixed.append(i)
    return out, np.array(fixed, dtype=int)


def load_kiva(drop_generation_loans: bool = True, realign: bool = True) -> Answers:
    """Kiva answers. By default the 40 training loans shown, with labels, to the LLM that wrote
    the questions are removed, because they cannot also be used to score those questions, and
    padded short replies are re-aligned (realign_short_replies)."""
    rows = [json.loads(line) for line in (KIVA / "answers.jsonl").read_text().splitlines() if line.strip()]
    if len({r["id"] for r in rows}) != len(rows) or any(len(r["scores"]) != 14 for r in rows):
        raise ValueError("Kiva answers must hold 14 scores for each of the loans, once")
    if drop_generation_loans:
        gen = {r["id"] for r in json.loads((KIVA / "generation_loans.json").read_text())}
        rows = [r for r in rows if r["id"] not in gen]
    ids = np.array([r["id"] for r in rows])
    scores = np.array([r["scores"] for r in rows], dtype=float)
    if realign:
        scores, _ = realign_short_replies(scores, kiva_facts(ids))
    questions = json.loads((KIVA / "questions.json").read_text())
    return Answers(X=(scores >= KIVA_YES).astype(np.int8).T.copy(), y=np.array([r["label"] for r in rows], dtype=np.int8),
                   questions=questions, source=np.array(["llm"] * len(questions)), ids=ids,
                   blank=np.isnan(scores).T.copy())


def common_founders(ids_a, ids_b):
    """Positions of the founders present in both id lists, in the order of `ids_a`."""
    pos_b = {h: j for j, h in enumerate(ids_b)}
    ia = np.array([i for i, h in enumerate(ids_a) if h in pos_b])
    return ia, np.array([pos_b[ids_a[i]] for i in ia])
