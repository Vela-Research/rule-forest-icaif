"""Re-ask Gemini for the Kiva loans whose reply the original scoring script padded.

The original run (data/kiva/answers.jsonl) padded a 13-score reply with its mean instead of
retrying, which left the later answers one question early (rule_forest.data.padded_rows finds
these 1,433 loans). This script asks again for those loans only, with the same model, prompt and
temperature, and retries any reply that does not hold exactly 14 scores. It writes
data/kiva/answers_rescored.jsonl and can be stopped and resumed. It is the only script in the
release that calls an LLM.

Needs `pip install google-genai pydantic`, GOOGLE_API_KEY or GEMINI_API_KEY in the environment,
and the Kiva records file with the loan descriptions (not shipped here):
  python experiments/kiva_rescore.py --records /path/to/kiva_records.jsonl
  python experiments/kiva_rescore.py --records ... --dry-run    # print one prompt, call nothing
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rule_forest import data

OUT = data.KIVA / "answers_rescored.jsonl"
MODEL, TEMPERATURE = "gemini-2.5-flash", 0.3

# The scoring prompt, verbatim from the original run (prompts/kiva.md).
SCORE_SYSTEM = (
    "You answer YES/NO diagnostic questions about a single Kiva microloan, judging "
    "only from the loan shown. For each question output a value in [0,1]: 1.0 = "
    "clearly yes, 0.0 = clearly no, intermediate = uncertain. Return strictly the "
    "requested JSON."
)


def meta(rec):
    return (f"sector: {rec.get('sector','')}\ncountry: {rec.get('country','')}\n"
            f"loan_amount_usd: {rec.get('loan_amount','')}")


def build_prompt(rec, questions):
    qs = "\n".join(f"[{i}] {q}" for i, q in enumerate(questions)); n = len(questions)
    return (f"LOAN METADATA\n{meta(rec)}\n\nLOAN NARRATIVE\n{(rec.get('description') or '')[:2000]}\n\n"
            f"QUESTIONS ({n})\n{qs}\n\n"
            f'Return JSON {{"scores": [s0, ..., s{n-1}]}} with {n} floats in [0,1], one per question in order.')


def parse_obj(resp, key):
    parsed = getattr(resp, "parsed", None)
    if parsed is not None and getattr(parsed, key, None) is not None:
        return getattr(parsed, key)
    txt = (resp.text or "").strip().strip("`")
    if txt[:4].lower() == "json":
        txt = txt[4:]
    a, b = txt.find("{"), txt.rfind("}")
    return json.loads(txt[a:b + 1])[key]


def score_one(client, types, Scores, rec, questions, retries=5):
    """One loan; a reply without exactly 14 scores is retried (temperature 0.3, then 0.6, 0.9, 1.0)."""
    prompt = build_prompt(rec, questions)
    temps = [TEMPERATURE, 0.6, 0.9, 1.0, 1.0]
    for a in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temps[a], response_mime_type="application/json", response_schema=Scores,
                    system_instruction=SCORE_SYSTEM, max_output_tokens=2048,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)))
            raw = parse_obj(resp, "scores")
            if not isinstance(raw, list) or len(raw) != len(questions):
                raise ValueError(f"bad length {len(raw) if isinstance(raw, list) else '?'}")
            return [max(0.0, min(1.0, float(x))) for x in raw], None, temps[a]
        except Exception as e:                                   # noqa: BLE001 (any API failure is retried)
            if a == retries - 1:
                return None, str(e)[:200], temps[a]
            time.sleep(min(2 ** a + 0.5, 20))
    return None, "exhausted", None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, help="kiva_records.jsonl with id, description, sector, country, loan_amount")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(line) for line in (data.KIVA / "answers.jsonl").read_text().splitlines() if line.strip()]
    todo_ids = [rows[i]["id"] for i in data.padded_rows(np.array([r["scores"] for r in rows]))]
    records = {r["id"]: r for r in (json.loads(x) for x in Path(a.records).read_text().splitlines() if x.strip())}
    questions = json.loads((data.KIVA / "questions.json").read_text())
    with open(data.KIVA / "loans.csv") as fh:
        loans = {r["id"]: r for r in csv.DictReader(fh)}
    for i in todo_ids:            # the records must describe the same loans as the shipped metadata
        r, m = records[i], loans[i]
        if (r["country"], r["sector"], float(r["loan_amount"])) != (m["country"], m["sector"], float(m["loan_amount"])):
            raise ValueError(f"records file disagrees with data/kiva/loans.csv for {i}")
    done = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                if o.get("scores") is not None:
                    done[o["id"]] = o
    todo = [records[i] for i in todo_ids if i not in done]
    print(f"{len(todo_ids)} padded loans, {len(done)} already re-asked, {len(todo)} to ask ({MODEL}, T={TEMPERATURE})")
    if a.dry_run:
        print("\n--- system ---\n" + SCORE_SYSTEM + "\n--- user (first loan) ---\n" + build_prompt(todo[0] if todo else records[todo_ids[0]], questions))
        return
    if not todo:
        return
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY.")
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class Scores(BaseModel):
        scores: list[float]

    client = genai.Client(api_key=key)
    lock, t0, n = threading.Lock(), time.time(), {"ok": 0, "err": 0}
    with open(OUT, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
        futures = {ex.submit(score_one, client, types, Scores, rec, questions): rec["id"] for rec in todo}
        for f in as_completed(futures):
            scores, err, temp = f.result()
            with lock:
                fh.write(json.dumps({"id": futures[f], "scores": scores, "error": err, "model": MODEL,
                                     "temperature": temp}) + "\n")
                fh.flush()
                n["err" if err else "ok"] += 1
                k = n["ok"] + n["err"]
                if k % 100 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)}  ok={n['ok']} err={n['err']}  [{time.time() - t0:.0f}s]", flush=True)
    if n["err"]:
        print(f"{n['err']} loans failed; run the same command again to retry them.")
    else:
        print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
