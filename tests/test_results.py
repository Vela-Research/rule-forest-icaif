"""The committed results hold the numbers the camera-ready paper reports (3 decimals).

If a change to the code moves one of these, the paper and results/PAPER_NUMBERS.md must move too.
"""
import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[1] / "results"


def load(name):
    return json.loads((R / f"{name}.json").read_text())


def close(x, v):
    assert round(x, 3) == pytest.approx(v, abs=1e-9), (x, v)


def test_table2():
    rows = load("table2")["rows"]
    expected = {"rule_forest": (0.103, 0.080, 0.097), "elastic_net": (0.112, 0.066, 0.098),
                "engineered_ml": (0.071, 0.055, 0.066), "embeddings": (0.121, 0.014, 0.047),
                "o3_few_shot": (0.077, 0.196, 0.088), "gpt_4o_mini_zero_shot": (0.079, 0.139, 0.087),
                "gptree": (0.068, 0.095, 0.071), "rrm": (0.080, 0.102, 0.082)}
    for k, (p, r, f) in expected.items():
        close(rows[k]["precision"], p); close(rows[k]["recall"], r); close(rows[k]["f05"], f)


def test_rule_forest_matches_the_elastic_net_within_noise():
    b = load("table2")["intervals"]["rule_forest_minus_elastic_net"]["f05"]
    close(b["delta"], -0.001); close(b["ci95"][0], -0.032); close(b["ci95"][1], 0.035)
    o3 = load("table2")["intervals"]["rule_forest_minus_o3_few_shot"]["f05"]
    close(o3["delta"], 0.010); close(o3["ci95"][0], -0.034); close(o3["ci95"][1], 0.055)


def test_table3():
    t = load("table3")
    expected = {"vote_hamming_0.15": 0.097, "vote_hamming_0.10": 0.088, "vote_hamming_0.20": 0.059,
                "vote_jaccard_0.90": 0.093, "vote_cosine_0.85": 0.094,
                "vote_claude_sonnet_4_5": 0.079, "vote_gpt_5_2": 0.048}
    for k, v in expected.items():
        close(t["vote"][k]["f05"], v)
    for k, v in {"en_gpt_4o_mini": 0.098, "en_claude_sonnet_4_5": 0.066, "en_gpt_5_2": 0.050}.items():
        close(t["elastic_net"][k]["f05"], v)


def test_expert():
    e = load("expert")
    close(e["expert_arm"]["f05"], 0.125); close(e["expert_arm"]["precision"], 0.164)
    close(e["elastic_net_58_questions"]["f05"], 0.102)


def test_shortlists():
    s = {(i["a"], i["b"], i["K"]): i for i in load("shortlists")["intervals"]}
    close(s[("vote_expert", "o3_few_shot", 60)]["delta"], 0.073)
    close(s[("vote_hamming_0.15", "o3_few_shot", 402)]["delta"], -0.006)


def test_figures():
    f4 = load("figure4")
    s4 = {r["successes"]: r for r in f4["slices"]["S4"]["table"]}
    close(s4[15]["gap"], 0.044); close(s4[158]["gap"], 0.004)
    assert f4["elastic_net_overfits_more_at_points"] == 18 and f4["points"] == 20
    f5 = {r["successes"]: r for r in load("figure5")["table"]}
    assert set(f5) == {15, 30, 60, 93, 130, 180, 202}
    close(f5[202]["gap"], -0.020)
    close(f5[93]["gap"], 0.017)


def test_predictions_load_without_pickle():
    import numpy as np
    files = sorted((R / "predictions").glob("*.npz"))
    assert len(files) == 13
    for p in files:
        z = np.load(p, allow_pickle=False)
        assert all(z[k].dtype != object for k in z.files)
