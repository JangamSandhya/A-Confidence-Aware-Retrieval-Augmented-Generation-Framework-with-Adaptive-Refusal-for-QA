

import logging
import math
import re
from typing import List, Optional, Tuple

import numpy as np

from config import RAGConfig
from ragtruth_corpus import Document

logger = logging.getLogger(__name__)




class TFIDFEncoder:
   

    def __init__(self, dim: int = 128):
        self.dim = dim
        self._vocab:  dict  = {}
        self._idf:    np.ndarray = np.array([])
        self._svd_Vt: np.ndarray = np.array([])
        self._fitted  = False


    def fit(self, texts: List[str]) -> "TFIDFEncoder":
        n = len(texts)
        tokenised = [self._tok(t) for t in texts]

       
        df: dict = {}
        for toks in tokenised:
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1

        self._vocab = {w: i for i, w in enumerate(sorted(df))}
        self._idf = np.array(
            [math.log((n + 1) / (df[w] + 1)) + 1.0 for w in sorted(df)],
            dtype=np.float32,
        )

        M = self._tfidf_matrix(tokenised)
        actual_dim = min(self.dim, M.shape[0] - 1, M.shape[1])
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        self._svd_Vt = Vt[:actual_dim]
        self._fitted = True
        logger.info(
            f"TFIDFEncoder fitted: vocab={len(self._vocab)}, "
            f"SVD dim={actual_dim}, docs={n}"
        )
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")
        tokenised = [self._tok(t) for t in texts]
        M = self._tfidf_matrix(tokenised)
        reduced = M @ self._svd_Vt.T            # (N, dim)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-9
        return (reduced / norms).astype(np.float32)


    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.findall(r"[a-z]{2,}", text.lower())

    def _tfidf_matrix(self, tokenised: List[List[str]]) -> np.ndarray:
        V = len(self._vocab)
        M = np.zeros((len(tokenised), V), dtype=np.float32)
        for i, toks in enumerate(tokenised):
            tf: dict = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            n = max(len(toks), 1)
            for tok, cnt in tf.items():
                j = self._vocab.get(tok)
                if j is not None:
                    M[i, j] = (cnt / n) * self._idf[j]
        return M




class DPREncoder:
    

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._ctx_tok  = None
        self._ctx_enc  = None
        self._q_tok    = None
        self._q_enc    = None
        self._available = self._try_load()

    def is_available(self) -> bool:
        return self._available

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        return self._encode_batch(texts, self._ctx_tok, self._ctx_enc)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self._encode_batch(texts, self._q_tok, self._q_enc)


    def _try_load(self) -> bool:
        try:
            import torch
            from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer,
                                      DPRQuestionEncoder, DPRQuestionEncoderTokenizer)
            dev = self.cfg.generator_device
            logger.info("Loading DPR encoders …")
            self._ctx_tok = DPRContextEncoderTokenizer.from_pretrained(self.cfg.dpr_ctx_encoder)
            self._ctx_enc = DPRContextEncoder.from_pretrained(self.cfg.dpr_ctx_encoder).eval()
            self._q_tok   = DPRQuestionEncoderTokenizer.from_pretrained(self.cfg.dpr_question_encoder)
            self._q_enc   = DPRQuestionEncoder.from_pretrained(self.cfg.dpr_question_encoder).eval()
            if dev != "cpu":
                self._ctx_enc = self._ctx_enc.to(dev)
                self._q_enc   = self._q_enc.to(dev)
            logger.info("DPR encoders ready.")
            return True
        except Exception as e:
            logger.warning(f"DPR unavailable ({e}). TF-IDF backend will be used.")
            return False

    def _encode_batch(self, texts, tokenizer, model) -> np.ndarray:
        import torch
        dev = self.cfg.generator_device
        all_embs = []
        bs = self.cfg.embedding_batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i: i + bs]
            inputs = tokenizer(
                batch, return_tensors="pt",
                truncation=True, max_length=self.cfg.max_input_length,
                padding="max_length",
            )
            if dev != "cpu":
                inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                emb = model(**inputs).pooler_output
            all_embs.append(emb.cpu().numpy().astype(np.float32))
        return np.vstack(all_embs)




class CorpusEncoder:
    

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        dpr = DPREncoder(cfg)
        if dpr.is_available():
            self._backend = "dpr"
            self._dpr = dpr
            self._tfidf: Optional[TFIDFEncoder] = None
        else:
            self._backend = "tfidf"
            self._dpr = None
            self._tfidf = TFIDFEncoder(dim=cfg.embedding_dim)

    @property
    def backend(self) -> str:
        return self._backend

    def encode_corpus(self, documents: List[Document]) -> np.ndarray:
        texts = [f"{d.question} {d.source_info}".strip() for d in documents]
        if self._backend == "dpr":
            return self._dpr.encode_docs(texts)
        self._tfidf.fit(texts)
        return self._tfidf.encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        if self._backend == "dpr":
            return self._dpr.encode_queries([query])[0]
        if not self._tfidf or not self._tfidf._fitted:
            raise RuntimeError("Encoder not fitted. Call encode_corpus() first.")
        return self._tfidf.encode([query])[0]
