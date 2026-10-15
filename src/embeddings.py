"""Frozen multilingual Transformer representations, cached by content and revision."""
from __future__ import annotations
import hashlib
import numpy as np
from src.common import ARTIFACTS, CONFIG

class Embedder:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(CONFIG["torch_cpu_threads"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(CONFIG["embedding_model"], revision=CONFIG["embedding_revision"], device=self.device)
        self.model.max_seq_length = 256

    def encode(self, texts, prefix="query: ", cache=True):
        texts = list(texts)
        key = hashlib.sha256((CONFIG["embedding_model"] + CONFIG["embedding_revision"] + prefix + "\0".join(texts)).encode()).hexdigest()
        path = ARTIFACTS / "cache" / f"{key}.npy"
        if cache and path.exists():
            return np.load(path, allow_pickle=False)
        vectors = self.model.encode([prefix+t for t in texts], batch_size=CONFIG["embedding_batch_size"],
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=len(texts)>200).astype("float32")
        if cache:
            path.parent.mkdir(parents=True,exist_ok=True); np.save(path,vectors)
        return vectors
