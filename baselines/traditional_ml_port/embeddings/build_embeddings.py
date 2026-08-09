"""Embeddings for the founder eval set: frozen all-MiniLM-L6-v2 sentence embeddings of the
anonymised founder prose. L2-normalised, 384-d. Deterministic (frozen encoder, eval mode).
Reproducible from the shipped 03_full_cross_validation_test_data_anonymised.csv alone.

  python build_embeddings.py     # -> embeddings_minilm.npy  (8204 x 384)
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REL = HERE.parents[2]
EVAL = REL / "rrf_submission" / "precomputed" / "03_full_cross_validation_test_data_anonymised.csv"
MODEL = "all-MiniLM-L6-v2"


def build_embeddings():
    from sentence_transformers import SentenceTransformer
    d = pd.read_csv(EVAL)
    model = SentenceTransformer(MODEL)
    emb = model.encode(d["summary"].astype(str).tolist(),
                       normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    return d, np.asarray(emb, dtype=np.float32)


if __name__ == "__main__":
    d, emb = build_embeddings()
    np.save(HERE / "embeddings_minilm.npy", emb)
    print(f"wrote embeddings_minilm.npy  shape={emb.shape}  (model={MODEL}, L2-normalised)")
