

import json
import logging
import os
from typing import List, Tuple

import numpy as np

from config import RAGConfig
from ragtruth_corpus import Document

logger = logging.getLogger(__name__)


RetrievedDoc = Tuple[Document, float]


class FAISSIndex:
    

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._index  = None
        self._docs:  List[Document] = []
        self._embs:  np.ndarray | None = None   # kept for numpy fallback
        self._use_faiss = self._check_faiss()


    def build(self, embeddings: np.ndarray, documents: List[Document]):
        assert len(embeddings) == len(documents), "Embeddings/docs count mismatch."
        self._docs = documents
        dim = embeddings.shape[1]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = (embeddings / norms).astype(np.float32)

        if self._use_faiss:
            self._index = self._build_faiss(embeddings, dim)
        else:
            logger.warning("FAISS not found; using numpy brute-force search.")
            self._embs = embeddings

        logger.info(f"Index built: {len(documents)} docs, dim={dim}, "
                    f"backend={'faiss' if self._use_faiss else 'numpy'}")

    def _build_faiss(self, embeddings: np.ndarray, dim: int):
        import faiss
        t = self.cfg.faiss_index_type.upper()
        if t == "FLAT":
            index = faiss.IndexFlatIP(dim)
        elif t == "IVF":
            q = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(q, dim, self.cfg.faiss_nlist,
                                        faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
        elif t == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown faiss_index_type: {t!r}")
        index.add(embeddings)
        return index


    def save(self, index_path: str, docs_path: str):
        for p in (index_path, docs_path):
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        if self._use_faiss and self._index is not None:
            import faiss
            faiss.write_index(self._index, index_path)
        else:
            np.save(index_path + ".npy", self._embs)
        with open(docs_path, "w") as f:
            json.dump([d.to_dict() for d in self._docs], f)
        logger.info(f"Index saved → {index_path}")

    def load(self, index_path: str, docs_path: str):
        with open(docs_path) as f:
            self._docs = [Document.from_dict(d) for d in json.load(f)]
        if self._use_faiss:
            import faiss
            self._index = faiss.read_index(index_path)
        else:
            self._embs = np.load(index_path + ".npy")
        logger.info(f"Index loaded: {len(self._docs)} docs.")


    def search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> Tuple[List[RetrievedDoc], np.ndarray]:
        
        qe = query_embedding.astype(np.float32)
        qe /= np.linalg.norm(qe) + 1e-9

        if self._use_faiss and self._index is not None:
            k_ext = min(top_k * 4, len(self._docs))
            ext_s, _ = self._index.search(qe.reshape(1, -1), k_ext)
            full_scores = ext_s[0]
            scores, idxs = self._index.search(qe.reshape(1, -1), top_k)
            scores, idxs = scores[0], idxs[0]
        else:
            full_scores = self._embs @ qe
            idxs   = np.argsort(-full_scores)[:top_k]
            scores = full_scores[idxs]

        results = [
            (self._docs[i], float(s))
            for i, s in zip(idxs, scores)
            if 0 <= i < len(self._docs)
        ]
        return results, full_scores

    def __len__(self):
        return len(self._docs)


    @staticmethod
    def _check_faiss() -> bool:
        try:
            import faiss  
            return True
        except ImportError:
            return False
