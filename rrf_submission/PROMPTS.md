# Rule Forest — question-generation prompt templates

These are the **exact** LLM prompt templates used to generate the YES/NO green-flag
questions that Rule Forest turns into binary features. They are reproduced verbatim
from `rrf_submission/src/llms/prompts.py` (the literal source is shipped alongside
this file).

The shipped question/response matrices in `rrf_submission/precomputed/` were produced
by the **anonymised** question-generation pipeline, which calls the two anonymised
templates below. `{num_questions}` and `{founder_summaries}` are filled in at call time
(questions are generated in batches over successive sets of founder summaries). That
generation pipeline needs an LLM API key and is not part of this offline release; the
literal prompt source is shipped in `rrf_submission/src/llms/prompts.py`.

---

## 1. Initial batch — `get_anonymised_question_prompt`  *(canonical; used for the paper)*

**System message**

> You are a VC analyst specializing in evaluating founders. Your task is to generate
> high-quality, objective YES/NO questions based on provided founder summaries. These
> questions should evaluate factors such as education, work history, leadership, and
> relevant experience.

**User prompt**

```
Generate {num_questions} YES/NO questions that can be answered using the anonymised founder summaries below.
Each question should be objective and specific, relating to observable traits in the summaries (e.g., education, roles, founding experience).
IMPORTANT FORMATTING REQUIREMENTS:
1. Return ONLY the questions, one per line
2. Do NOT include any introductory text or explanations
3. Do NOT include any numbering or bullet points
4. Each line should contain exactly one question that can be answered with Yes or No

Example format (DO NOT USE THIS QUESTION):
Has the founder previously held a senior leadership role?

Founder Summaries:

{founder_summaries}
```

---

## 2. Subsequent batches — `get_anonymised_updated_question_prompt`  *(canonical; used for the paper)*

**System message**

> You are a VC analyst specializing in evaluating startup founders. Your task is to
> generate objective YES/NO questions based on the anonymised founder summaries below.
> These questions should focus on traits like education, leadership, experience, and
> founding history. Each question must be answerable with 'Yes' or 'No' based solely on
> the summaries provided.

**User prompt**

```
We are now evaluating a new batch of anonymised founders. Please generate {num_questions} new objective YES/NO questions inspired by this batch.
Each question must be specific, verifiable from the data, and helpful for distinguishing successful from unsuccessful founders.

IMPORTANT FORMATTING REQUIREMENTS:
1. Return ONLY the questions, one per line.
2. Do NOT include any introductory text, explanations, or conversational filler.
3. Do NOT include any numbering or bullet points.
4. Each line must contain exactly one question that can be answered with 'Yes' or 'No'.

Example format (DO NOT USE THESE QUESTIONS):
Has the founder held a leadership role in a technology company?
Did the founder complete postgraduate education?

New Founder Summaries:

{founder_summaries}
```

---

## Non-anonymised variants (for completeness)

The same two-stage prompt was developed on the raw (non-anonymised) founder profiles
via `get_standard_question_prompt` / `get_standard_updated_prompt` in
`src/llms/prompts.py`. They are structurally identical to the anonymised templates
above; the release ships the anonymised pipeline, so the anonymised variants are the
ones that produced every number in the paper.
