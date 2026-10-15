"""Normalized cosine search with FAISS, plus the course's BM25/RRF improvement."""
from __future__ import annotations
import re
import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from src.common import DATA, load_kb

def tokenize(text):
    return re.findall(r"\w+",text.casefold())

class Retriever:
    def __init__(self,embedder):
        self.embedder = embedder
        self.history = pd.read_csv(DATA/"history.csv").fillna("").to_dict("records")
        self.kb = load_kb()
        self.ktexts = [f"{c['title']} > {c['section']}\n{c['text']}" for c in self.kb]
        self.hvectors = embedder.encode([c["text"] for c in self.history],prefix="passage: ")
        self.kvectors = embedder.encode(self.ktexts,prefix="passage: ")
        self.hindex = faiss.IndexFlatIP(self.hvectors.shape[1]);self.hindex.add(self.hvectors)
        self.kindex = faiss.IndexFlatIP(self.kvectors.shape[1]);self.kindex.add(self.kvectors)
        self.bm25 = BM25Okapi([tokenize(t) for t in self.ktexts])

    def search_kb(self,text,q=None,k=3,method="dense"):
        q = self.embedder.encode([text],cache=False) if q is None else np.asarray(q,dtype="float32").reshape(1,-1)
        scores,idx = self.kindex.search(q,len(self.kb))
        dense_order = idx[0].tolist()
        cosine = dict(zip(dense_order,scores[0].tolist()))
        if method == "dense":
            order = dense_order
        else:
            sparse = self.bm25.get_scores(tokenize(text))
            # Arabic/EN share no lexical terms. With no BM25 evidence, preserve dense order.
            if float(sparse.max()) <= 0:
                order = dense_order
            else:
                sparse_order = np.argsort(-sparse).tolist()
                fused = np.zeros(len(self.kb))
                for ranking in [dense_order,sparse_order]:
                    for rank,i in enumerate(ranking,start=1):
                        fused[i] += 1/(60+rank)
                order = np.argsort(-fused).tolist()
        return [{**self.kb[i],"score":round(cosine[i],5)} for i in order[:k]]

    def search_history(self,text,q=None,k=3):
        q = self.embedder.encode([text],cache=False) if q is None else np.asarray(q,dtype="float32").reshape(1,-1)
        scores,idx = self.hindex.search(q,min(k,len(self.history)))
        return [{"ticket_id":self.history[i]["ticket_id"],"text":self.history[i]["text"],
            "intent":self.history[i]["intent"],"response":self.history[i]["response"],"score":round(float(score),5)} for i,score in zip(idx[0],scores[0])]
