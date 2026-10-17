# Data

Everything the experiments read. No file here is written by the code.

## Founders

The 9,192 founders lead U.S. companies founded between 2010 and 2016 that had raised between $100K
and $4M at the time of the investment decision. Each is described by an anonymised
natural-language profile. A founder counts as a success when the company later reached an IPO or
acquisition above $500M, or raised more than $500M in total.

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
  GPT-4o-mini for the same founders (12 of its blank cells are founders who were not asked). It has
  no label row; the labels are those of `gpt_4o_mini.csv`.
- `gpt_4o_mini_reasked_blanks.csv` holds 289 of the 417 blank GPT-4o-mini cells, re-asked later
  with the same prompt (GPT-4o-mini, temperature 0); 102 came back YES. Only the sensitivity check
  in `experiments/founders.py` uses them.
- Founder ids are the first 5 characters of the anonymised id. 27 pairs of evaluation founders
  share an id and appear once, and the answering run never sent the last 4 evaluation founders,
  so the GPT-4o-mini matrix has 8,204 - 27 - 4 = 8,173 founders. In one pair (id `72c43`) one
  founder succeeded and the other did not; the matrix labels it a success, and which founder's
  answers it holds is not recorded.
- 50 founder ids in the Claude Sonnet 4.5 and GPT-5.2 files match no id in `profiles/` (none of
  them is a success). Each generator is scored on its own file, and comparisons between
  generators use the 8,117 founders common to both files.
- The files produced by the original pipeline also carried 13 summary columns (precision, F0.5
  and so on), computed from the labels of all evaluation founders. They are removed here because
  no analysis may use them; the original files are in the git tag `icaif26-submission`.

### `founders/prompting/`

These are the direct-prompting baselines, in which an LLM is asked whether each founder will succeed. Row `Prediction` is
`1` for "Successful". There are six runs, of o3, o3-mini and GPT-4o-mini, each zero-shot and few-shot. The
o3-mini few-shot run covers only 500 founders. Founders a run did not score count as not flagged.

### `founders/question_chain/`

These files trace how the 50 GPT-4o-mini questions were reached, through `generated.csv` (240 questions as written),
`generated_exact_unique.csv` (132), `after_semantic_dedup.csv` (62) and `validation_answers.csv`
(the 62 questions answered for 479 validation founders, 49 of them successes, in the same format as
`answers/`). The screen kept questions with validation precision and recall of at least 0.10.
Recomputed from this file, 51 questions pass, while 50 were scored on the evaluation founders;
`results/questions.json` lists the three questions on which the two disagree.

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

- **Generation loans.** The question-writing script drew 20 expired and 20 funded loans from its
  training split with `random.Random(42)` (`prompts/kiva.md`). `split` in `answers.jsonl` is that
  run's split; the experiment code's records file was later rebuilt with a different one. Applied
  to the records in their original order with this split, the sampler gives the 40 loans listed
  here, and every one of the 11 countries named in question 6 appears among their 20 funded loans
  (a test checks this). Every experiment removes them, leaving 4,960 loans and 202 positives (4.1%).
- **Short replies.** For at least 1,433 loans the LLM returned 13 scores, and the scoring script
  padded the reply with its own mean, so every answer after the skipped question sits one question
  early. These rows are recognisable because their last score equals the mean of the first 13
  (strictly between 0 and 1). `rule_forest.data.load_kiva` re-aligns them using the loan's facts,
  not its label. It finds the skipped positions whose re-aligned answers best match the facts and
  keeps an answer only where all of them put the same answer on that question; every other cell,
  including the skipped question, is left blank (NO). This turns 83 of the 2,885 YES answers in
  these rows into NO.
- **How well it works.** The answers of other loans match the facts in 98.7% of cells. The padded
  rows match in 89.5% as returned. Choosing the position from half the questions and checking the
  other half, they match in 96.7% to 98.1% after re-alignment.
- **Why it matters.** Padded replies are not random. 2.7% of padded loans expired, against 5.1% of
  the others, so the shifted answers carried a label-linked artefact into the as-returned data.
  About 150 complete 14-score replies also look shifted, by the LLM itself; they cannot be detected
  from the reply and are left as returned. `python experiments/figure5.py run --as-scored` and then
  `plot --as-scored` recompute Figure 5's numbers without the repair (`results/figure5_as_scored.json`).

The prompts that produced every file are in `prompts/`.
