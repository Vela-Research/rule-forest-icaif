# Data

Everything the experiments read. No file here is written by the code.

## Founders

The 9,192 founders lead U.S. companies founded between 2010 and 2016. Each is described by an
anonymised natural-language profile. Labels follow the definition of VCBench
([arXiv:2509.14448](https://arxiv.org/abs/2509.14448), Section 3): a founder is successful "if the
company was acquired or had an initial public offering (IPO) above $500M valuation, or raised more
than $500M in funding", and "companies that raised $100K–$4M at inception but did not achieve an
exit, IPO, or substantial follow-on funding within eight years of founding are classified as
unsuccessful".

### `founders/profiles/`

| File | Founders | Successes | Role |
|---|---:|---:|---|
| `generation.csv` | 497 | 248 | Shown to the LLM to write the questions (profiles only, never labels) |
| `validation.csv` | 491 | 50 | Used once, to screen out weak questions before scoring every founder |
| `evaluation.csv` | 8,204 | 158 | Every reported result |

Columns are `founder_uuid` (an anonymised id), `summary` (the profile) and `startup_success`.

### `founders/answers/`

One file per question generator. Rows are questions, in the order they were scored, and the
first row, `Success`, holds the labels. Columns are evaluation founders, in the order every
cross-validation split uses. A cell is `1` (YES), `0` (NO) or blank (the reply was missing or
could not be read). Every analysis counts a blank as NO.

| File | Questions | Founders | Successes | Blank cells |
|---|---:|---:|---:|---:|
| `gpt_4o_mini.csv` | 50 | 8,173 | 158 | 417 (0.10%) |
| `claude_sonnet_4_5.csv` | 22 | 8,171 | 157 | 1,988 (1.1%) |
| `gpt_5_2.csv` | 21 | 8,171 | 157 | 1,622 (0.95%) |
| `expert.csv` | 8 | 8,173 | | 61 |

- `expert.csv` holds the 8 questions written by a venture-capital partner, answered by
  GPT-4o-mini for the same founders. It has no label row; the labels are those of `gpt_4o_mini.csv`.
- `gpt_4o_mini_reasked_blanks.csv` holds 289 of the 417 blank GPT-4o-mini cells, re-asked later
  with the same prompt (GPT-4o-mini, temperature 0); 102 came back YES. Only the sensitivity check
  in `experiments/founders.py` uses them.
- Founder ids are the first 5 characters of the anonymised id. 27 pairs of evaluation founders
  share an id and appear once, and the answering run never sent the last 4 evaluation founders,
  so the GPT-4o-mini matrix has 8,204 - 27 - 4 = 8,173 founders.
- Each generator is scored on its own file, and comparisons between generators use the 8,117
  founders common to both files.

### `founders/prompting/`

These are the direct-prompting baselines, in which an LLM is asked whether each founder will succeed. Row `Prediction` is
`1` for "Successful". There are six runs, of o3, o3-mini and GPT-4o-mini, each zero-shot and few-shot. The
o3-mini few-shot run covers only 500 founders. Founders a run did not score count as not flagged.

### `founders/question_chain/`

These files trace how the 50 GPT-4o-mini questions were reached, through `generated.csv` (240 questions as written),
`generated_exact_unique.csv` (132), `after_semantic_dedup.csv` (62) and `validation_answers.csv`
(the 62 questions answered for 479 validation founders, 49 of them successes, in the same format as
`answers/`). The screen kept questions with validation precision and recall of at least 0.10;
50 were scored on the evaluation founders.

## Kiva

5,000 Kiva microloans with an English description that reached a funding decision, sampled from
a public Kiva snapshot. A loan counts as positive when it expired without being fully funded
(222 loans, 4.4%).

| File | Contents |
|---|---|
| `kiva/questions.json` | The 14 YES/NO questions, written by Gemini 2.5 Flash |
| `kiva/answers.jsonl` | The scores as returned. One record per loan with `id`, `label`, `split`, and `scores`, a value in [0, 1] for each question (0.5 or more counts as YES) |
| `kiva/loans.csv` | Each loan's country, sector and amount, from the same snapshot |
| `kiva/question_facts.json` | The fact each question asks about (for example, `country == Kenya`) |
| `kiva/generation_loans.json` | The 40 training loans (20 expired, 20 funded) shown, with labels, to the LLM that wrote the questions |

- **Generation loans.** The 40 loans in `generation_loans.json` are removed from every experiment,
  leaving 4,960 loans and 202 positives (4.1%).
- **Short replies.** For at least 1,433 loans the LLM returned 13 scores and the scoring script
  padded the reply with their mean, so every answer after the skipped question sits one question
  early. `rule_forest.data.load_kiva` re-aligns these rows using each loan's facts, never its label,
  and leaves an answer blank (NO) where its position is ambiguous. Checked on held-out questions,
  the re-aligned rows match the loan facts in 96.7% to 98.1% of cells, against 98.7% for other
  loans. `python experiments/figure5.py run --as-scored` and then `plot --as-scored` recompute
  Figure 5 without the re-alignment (`results/figure5_as_scored.json`).

The prompts that produced every file are in `prompts/`.
