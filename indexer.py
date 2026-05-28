

import json
import logging
import os
import numpy as np
from typing import List, Tuple, Optional

from config import RAGConfig
from corpus import Document

logger = logging.getLogger(__name__)

RetrievedDoc = Tuple[Document, float]   


class FAISSIndexer:

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._index = None
        self._documents: List[Document] = []
        self._embeddings: Optional[np.ndarray] = None   
        self._use_faiss = self._check_faiss()


    def build(self, embeddings: np.ndarray, documents: List[Document]):
        assert len(embeddings) == len(documents), "Embeddings / docs count mismatch."
        self._documents = documents
        dim = embeddings.shape[1]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = (embeddings / norms).astype(np.float32)

        if self._use_faiss:
            self._index = self._build_faiss(embeddings, dim)
        else:
            logger.warning("FAISS not available; using numpy brute-force search.")
            self._embeddings = embeddings

        logger.info(f"Index built: {len(documents)} documents, dim={dim}.")

    def _build_faiss(self, embeddings: np.ndarray, dim: int):
        import faiss
        idx_type = self.cfg.faiss_index_type.upper()
        if idx_type == "FLAT":
            index = faiss.IndexFlatIP(dim)
        elif idx_type == "IVF":
            quantiser = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantiser, dim, self.cfg.faiss_nlist,
                                        faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
        elif idx_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown faiss_index_type: {idx_type}")
        index.add(embeddings)
        return index


    def save(self, index_path: str, docs_path: str):
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(docs_path) or ".", exist_ok=True)
        if self._use_faiss and self._index is not None:
            import faiss
            faiss.write_index(self._index, index_path)
        else:
            np.save(index_path + ".npy", self._embeddings)
        with open(docs_path, "w") as f:
            json.dump([d.to_dict() for d in self._documents], f)
        logger.info(f"Index saved → {index_path}")

    def load(self, index_path: str, docs_path: str):
        with open(docs_path) as f:
            self._documents = [Document.from_dict(d) for d in json.load(f)]
        if self._use_faiss:
            import faiss
            self._index = faiss.read_index(index_path)
        else:
            self._embeddings = np.load(index_path + ".npy")
        logger.info(f"Index loaded: {len(self._documents)} documents.")


    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievedDoc]:
        qe = query_embedding.astype(np.float32)
        qe = qe / (np.linalg.norm(qe) + 1e-9)

        if self._use_faiss and self._index is not None:
            scores, idxs = self._index.search(qe.reshape(1, -1), top_k)
            scores, idxs = scores[0], idxs[0]
        else:
            scores_all = self._embeddings @ qe
            idxs = np.argsort(-scores_all)[:top_k]
            scores = scores_all[idxs]

        results = []
        for idx, score in zip(idxs, scores):
            if 0 <= idx < len(self._documents):
                results.append((self._documents[idx], float(score)))
        return results

    def search_with_all_scores(
        self, query_embedding: np.ndarray, top_k: int
    ) -> Tuple[List[RetrievedDoc], np.ndarray]:
        qe = query_embedding.astype(np.float32)
        qe = qe / (np.linalg.norm(qe) + 1e-9)

        if self._use_faiss and self._index is not None:
            k2 = min(top_k * 4, len(self._documents))
            scores_ext, _ = self._index.search(qe.reshape(1, -1), k2)
            raw_scores = scores_ext[0]
            scores, idxs = self._index.search(qe.reshape(1, -1), top_k)
            scores, idxs = scores[0], idxs[0]
        else:
            raw_scores = self._embeddings @ qe
            idxs = np.argsort(-raw_scores)[:top_k]
            scores = raw_scores[idxs]

        results = [
            (self._documents[i], float(s))
            for i, s in zip(idxs, scores)
            if 0 <= i < len(self._documents)
        ]
        return results, raw_scores


    @staticmethod
    def _check_faiss() -> bool:
        try:
            import faiss  # noqa
            return True
        except ImportError:
            return False

    def __len__(self):
        return len(self._documents)
