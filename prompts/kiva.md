# Kiva prompts

The Kiva experiment (Figure 5) uses its own, lighter question step. The code below is copied
verbatim from the script as it ran (file `score_questions.py` in the experiment code, which is not
part of this release).

1. **Writing the questions.** One Gemini 2.5 Flash call (temperature 0.4) shows 40 labelled
   training loans, 20 that expired unfunded and 20 that were funded, and asks for 14 YES/NO
   questions. Unlike the founder prompts, this prompt shows the labels. The 40 loans are listed
   in `data/kiva/generation_loans.json` and are removed before any evaluation.
2. **Answering.** For each loan, one Gemini 2.5 Flash call (temperature 0.3, rising on a retry)
   scores all 14 questions from 0 to 1. A score of 0.5 or more counts as YES. The scores are in
   `data/kiva/answers.jsonl` and the questions in `data/kiva/questions.json`.

When a reply held too few or too many scores, `score_one` below padded or cut it instead of
retrying; `data/README.md` describes how the padded replies are re-aligned.

```python
GEN_SYSTEM = (
    "You design YES/NO diagnostic questions that discriminate Kiva microloans that "
    "EXPIRE unfunded from those that reach their funding goal. Questions must be "
    "answerable purely from a loan's narrative and terms (sector, country, loan "
    "amount). Make them concrete, non-redundant, and predictive of funding failure; "
    "avoid questions about lender behaviour or outcomes not visible from the loan itself."
)


SCORE_SYSTEM = (
    "You answer YES/NO diagnostic questions about a single Kiva microloan, judging "
    "only from the loan shown. For each question output a value in [0,1]: 1.0 = "
    "clearly yes, 0.0 = clearly no, intermediate = uncertain. Return strictly the "
    "requested JSON."
)


def meta(rec):
    return (f"sector: {rec.get('sector','')}\ncountry: {rec.get('country','')}\n"
            f"loan_amount_usd: {rec.get('loan_amount','')}")


def generate_questions(client, model, n_q, records):
    rng = random.Random(42)
    train = [r for r in records if r["split"] == "train"]
    pos = [r for r in train if r["label"] == 1]; neg = [r for r in train if r["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    sample = pos[:20] + neg[:20]; rng.shuffle(sample)
    block = "\n\n".join(f"[{'EXPIRED' if r['label'] else 'FUNDED'}] {meta(r)}\n{(r.get('description') or '')[:400]}"
                        for r in sample)
    user = (f"Here are {len(sample)} labelled example loans:\n\n{block}\n\n"
            f"Write exactly {n_q} YES/NO diagnostic questions that best discriminate "
            f'loans that EXPIRE from those that FUND. Return JSON {{"questions": [...]}}.')
    resp = client.models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(
            temperature=0.4, response_mime_type="application/json", response_schema=Questions,
            system_instruction=GEN_SYSTEM, max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0)))
    qs = parse_obj(resp, "questions")
    if not qs:
        raise RuntimeError("no questions generated")
    return qs


def build_prompt(rec, questions):
    qs = "\n".join(f"[{i}] {q}" for i, q in enumerate(questions)); n = len(questions)
    return (f"LOAN METADATA\n{meta(rec)}\n\nLOAN NARRATIVE\n{(rec.get('description') or '')[:2000]}\n\n"
            f"QUESTIONS ({n})\n{qs}\n\n"
            f'Return JSON {{"scores": [s0, ..., s{n-1}]}} with {n} floats in [0,1], one per question in order.')


def score_one(client, model, temp, rec, questions, n_q, retries=5):
    prompt = build_prompt(rec, questions)
    temps = [temp, 0.6, 0.9, 1.0, 1.0]
    for a in range(retries):
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temps[min(a, len(temps) - 1)], response_mime_type="application/json",
                    response_schema=Scores, system_instruction=SCORE_SYSTEM, max_output_tokens=2048,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)))
            raw = parse_obj(resp, "scores")
            if not isinstance(raw, list) or abs(len(raw) - n_q) > 2:
                raise ValueError(f"bad length {len(raw) if isinstance(raw, list) else '?'}")
            vals = [max(0.0, min(1.0, float(x))) for x in raw]
            if len(vals) > n_q:
                vals = vals[:n_q]
            elif len(vals) < n_q:
                vals = vals + [sum(vals) / len(vals) if vals else 0.5] * (n_q - len(vals))
            return vals, None
        except Exception as e:
            if a == retries - 1:
                return None, str(e)[:120]
            time.sleep(min(2 ** a + 0.5, 20))
    return None, "exhausted"
```
