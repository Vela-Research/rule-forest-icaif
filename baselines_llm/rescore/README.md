# GPTree & RRM — Table 2 reproduction (offline re-scoring)

Reproduces the paper's Table 2 **GPTree** and **RRM** rows from each method's frozen
LLM-generated artifacts, with **no LLM / API calls**.

```bash
pip install numpy pandas scikit-learn
python rescore_gptree_rrm.py
```

## Why a re-scoring step

Both methods induce a predictor from labelled data with an LLM (GPTree builds a decision tree; RRM
mines a rule set). To compare them to Rule Forest on equal footing, we reuse each method's frozen
artifact and score it under the **same 10x10 nested cross-validation** used for Rule Forest,
re-fitting only the data-dependent part inside each fold and tuning the decision threshold per fold
at the true 1.9% operating prevalence:

- **GPTree** (`data/gptree/gptree_seed{42,43,44}_leaves.csv`): each founder's leaf assignment from the
  fixed tree. Per fold, `P(success | leaf)` is refit on the training founders; the threshold is tuned
  per fold.
- **RRM** (`data/rrm/rrm_seed{42,43,44}_scores.json`): each founder's frozen rule-based score. The
  rule set is not changed; only the threshold is tuned per fold.
- Labels: `data/eval_labels.json` (8,204 evaluation founders, 158 positive, 1.93%).

`F0.5 = 1.25*p*r/(0.25*p+r)`; the table reports **mean ± SD over the 100 folds, pooled across the 3
generation seeds** (which is why the SDs are large).

## Expected output vs the paper

| Method | Precision | Recall | F0.5 | Paper (Table 2) |
|---|---|---|---|---|
| GPTree | 0.066 | 0.095 | 0.068 ± 0.054 | 0.064 / 0.094 / 0.066 ± 0.054 |
| RRM    | 0.079 | 0.102 | 0.080 ± 0.060 | 0.074 / 0.095 / 0.076 ± 0.063 |

Precision, recall and the standard deviations match; the F0.5 point estimate differs by ~0.002-0.004,
which is under 5% of one cross-validation standard deviation (Monte-Carlo noise of the fold shuffles).

Note: the older single-held-out-split protocol (`baselines_llm/{gptree,rrm}/*_seed*_results.json`)
reports a different, lower headline (F0.5 ~0.040) because it tunes the threshold on a higher-prevalence
validation split rather than per fold at the true prevalence; the paper uses the 10x10 re-scoring here.
