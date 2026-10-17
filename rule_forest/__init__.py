"""Rule Forest: an equal-weight vote over LLM-written YES/NO questions.

Answers are a (questions x examples) 0/1 matrix X with labels y. Typical use:

    from rule_forest import data, vote
    a = data.load_founders()                       # the paper's 50 GPT-4o-mini questions
    fit = vote.fit(a.X, a.y, train=range(a.X.shape[1]))
    print(fit.scorecard(a.questions))
    flagged = fit.predict(a.X)

Modules: data (loaders), pool (candidate pool), vote (the vote and its nested cross-validation),
elastic_net (the learned-weight baseline), metrics (pooled scores and the paired bootstrap).
"""
