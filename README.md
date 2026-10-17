# Rule Forest

Code and data for *Rule Forest: Interpretable Ensembles of LLM-Generated Questions for
Rare-Outcome Prediction* (ACM ICAIF 2026).

Rule Forest asks an LLM to write YES/NO questions about a profile, answers them with an LLM, and
predicts success when at least T of the N best questions are answered YES. Every number and
figure in the paper is rebuilt here from the shipped LLM answers, with no API key and no network.

## Install

Tested with Python 3.13.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # under a minute
```

## Reproduce the paper

```bash
python reproduce.py --quick   # every table, figure and quoted number, from the committed model predictions (minutes)
python reproduce.py           # also refit every model (about 25 CPU-hours; about 2 hours on 16 cores)
```

Either way the results land in `results/` and the figures in `figures/`, and
[`results/PAPER_NUMBERS.md`](results/PAPER_NUMBERS.md) lists every number the paper quotes with
the file it comes from. `git status results figures` then shows whether anything changed.

| Paper | Script | Result |
|---|---|---|
| Table 2, Section 5.1 | `experiments/founders.py` | `results/table2.json` |
| Table 2, GPTree and RRM | `baselines/gptree_rrm/rescore.py` | `results/gptree_rrm.json` |
| Table 2, Engineered ML and Embeddings | `baselines/engineered_ml/run.py`, `baselines/embeddings/run.py` | `results/engineered_ml.json`, `results/embeddings.json` |
| Table 3, Section 5.3 | `experiments/founders.py` | `results/table3.json` |
| Section 5.2, expert questions and equal-length shortlists | `experiments/founders.py` | `results/expert.json`, `results/shortlists.json` |
| Sections 4.1 and 4.2, question counts and missing answers | `experiments/questions.py` | `results/questions.json` |
| Figure 4 | `experiments/figure4.py` | `results/figure4.json`, `figures/figure4.pdf` |
| Figure 5 | `experiments/figure5.py` | `results/figure5.json`, `figures/figure5.pdf` |

## How the evaluation works

- **Cross-validation.** Every trained model is scored by 10 repeats of stratified 10-fold
  cross-validation (seed 42 + repeat - 1), with the same folds for every model on the same data.
- **No test label is used for training.** In each outer training fold, Rule Forest builds its
  candidate pool (precision floor and removal of near-duplicate questions) and ranks the questions
  by training F0.5. It then picks N and T with inner folds of the training founders (10 for the
  tables, 5 in the Figure 4 and 5 sweeps), which score every (N, T) on that fixed ranking; the pool
  and the ranking are not refitted inside the inner folds. The elastic net tunes its penalty by inner cross-validation and its threshold on the
  training fold.
- **Pooled scoring.** Within each repeat the hits of the 10 test folds are pooled, and precision,
  recall and F0.5 are computed once. Tables report the mean and SD over the 10 repeats (over 30 for
  GPTree and RRM, whose three runs each have 10).
- **Intervals.** Differences between two models come with a paired bootstrap over founders
  (10,000 resamples, seed 42).

## Layout

```
rule_forest/     the method (data loaders, candidate pool, vote, elastic net, metrics)
experiments/     one script per part of the paper; each writes results/<name>.json
baselines/       GPTree, RRM, Engineered ML and Embeddings (see baselines/README.md)
data/            the founder and Kiva answers, profiles and labels (see data/README.md)
prompts/         every LLM prompt, verbatim
results/         numbers, model predictions (results/predictions/) and sweep draws
figures/         Figures 4 and 5
tests/           unit tests and checks of the reported numbers
```

## Use Rule Forest on your own data

Answers are a 0/1 matrix with one row per question and one column per example.

```python
import numpy as np
from rule_forest import vote

# X: (questions, examples) 0/1 answers; y: 0/1 labels; questions: list of question text
fit = vote.fit(X, y, train=np.arange(X.shape[1]))
print(fit.scorecard(questions))     # "Predict success when at least T of these N questions ..."
flagged = fit.predict(X_new)        # X_new has the same question rows
```

`vote.fit` takes the pool settings as keywords, for example `similarity="jaccard"`,
`threshold=0.9` or `floor=0.05`. The precision floor (default 0.02001) sits just above the
founders' 1.93% base rate and should be set near the base rate of your data.

## Versions

The version linked from the submitted paper is the git tag `icaif26-submission`. The camera-ready
version differs in five ways, each of which changes some reported numbers.

1. The candidate pool is rebuilt inside each training fold. Before, its precision floor and its
   choice between near-duplicates used the labels of all evaluation founders.
2. The elastic net, GPTree and RRM rows use the same pooled scoring as Rule Forest, and the
   elastic net reads each question once (it had read every GPT-4o-mini question twice).
3. The Jaccard and cosine settings of Table 3 now compute what their names say. Jaccard compared
   whole answer vectors, which made it a stricter Hamming rule, and the cosine clustering was given
   a square distance matrix where scipy expects a condensed one.
4. Figure 4 uses evaluation founders only.
5. Figure 5 leaves out the 40 loans the LLM saw when writing the Kiva questions, and puts back in
   place the answers of 1,433 Kiva replies that skipped a question (see `data/README.md`).

Two smaller changes move a few numbers by at most 0.001. Ties in the question ranking are now
broken in row order, the same on every machine. Equal-length shortlists now hold exactly K
founders; before, rounding each fold's share gave 400 instead of 402.

## License

The code is under the MIT license (see `LICENSE`). The data in `data/` are released to reproduce
the paper. The founder profiles are anonymised summaries built from licensed company and
professional-background data (paper, Section 3), and the Kiva loans come from Kiva's public data
snapshot.
