

import logging
import os
import numpy as np
from typing import List, Optional

from config import RAGConfig
from corpus import Document

logger = logging.getLogger(__name__)


class DPRRetriever:
    

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._backend = None         
        self._q_tokenizer = None
        self._q_encoder = None
        self._ctx_tokenizer = None
        self._ctx_encoder = None
        self._tfidf = None
        self._svd = None
        self._vocab = None
        self._idf = None
        self._device = cfg.generator_device
        self._try_load_dpr()


    def encode_question(self, question: str) -> np.ndarray:
        if self._backend == "dpr":
            return self._dpr_encode_question(question)
        return self._tfidf_encode([question])[0]

    def encode_corpus(self, documents: List[Document]) -> np.ndarray:
        if self._backend == "dpr":
            return self._dpr_encode_corpus(documents)
        texts = [f"{d.title} {d.text}" for d in documents]
        return self._tfidf_encode(texts)


    def _try_load_dpr(self):
        try:
            import torch
            from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
            from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
            logger.info("Loading DPR encoders from HuggingFace…")
            self._q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                self.cfg.dpr_question_encoder)
            self._q_encoder = DPRQuestionEncoder.from_pretrained(
                self.cfg.dpr_question_encoder).eval()
            self._ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                self.cfg.dpr_ctx_encoder)
            self._ctx_encoder = DPRContextEncoder.from_pretrained(
                self.cfg.dpr_ctx_encoder).eval()
            if self._device != "cpu":
                self._q_encoder = self._q_encoder.to(self._device)
                self._ctx_encoder = self._ctx_encoder.to(self._device)
            self._backend = "dpr"
            logger.info("DPR backend ready.")
        except Exception as e:
            logger.warning(f"DPR unavailable ({e}). Using TF-IDF fallback.")
            self._backend = "tfidf"
            self.cfg.embedding_dim = 256   # override dim for TF-IDF/SVD

    def _dpr_encode_question(self, question: str) -> np.ndarray:
        import torch
        inputs = self._q_tokenizer(
            question, return_tensors="pt", truncation=True,
            max_length=self.cfg.max_input_length, padding="max_length"
        )
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self._q_encoder(**inputs).pooler_output
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def _dpr_encode_corpus(self, documents: List[Document]) -> np.ndarray:
        import torch
        all_embs = []
        bs = self.cfg.embedding_batch_size
        for i in range(0, len(documents), bs):
            batch = documents[i: i + bs]
            texts = [f"{d.title} {d.text}" for d in batch]
            inputs = self._ctx_tokenizer(
                texts, return_tensors="pt", truncation=True,
                max_length=self.cfg.max_input_length, padding="max_length"
            )
            if self._device != "cpu":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self._ctx_encoder(**inputs).pooler_output
            all_embs.append(emb.cpu().numpy().astype(np.float32))
            logger.info(f"  Encoded batch {i//bs + 1} / {-(-len(documents)//bs)}")
        return np.vstack(all_embs)


    def _tfidf_encode(self, texts: List[str]) -> np.ndarray:
        if self._vocab is None:
            raise RuntimeError("Call _fit_tfidf() before encoding.")
        matrix = self._tfidf_matrix(texts)
        reduced = matrix @ self._svd_components.T
        norms = np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-9
        return (reduced / norms).astype(np.float32)

    def _fit_tfidf(self, texts: List[str]):
        import re, math
        tokenise = lambda t: re.findall(r"[a-z]+", t.lower())
        df = {}
        n = len(texts)
        tokenised = [tokenise(t) for t in texts]
        for tokens in tokenised:
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1
        self._vocab = {w: i for i, w in enumerate(df)}
        self._idf = np.array(
            [math.log((n + 1) / (df[w] + 1)) + 1 for w in self._vocab], dtype=np.float32
        )
        matrix = self._tfidf_matrix(texts, tokenised=tokenised)
        dim = min(self.cfg.embedding_dim, matrix.shape[1])
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        self._svd_components = Vt[:dim]
        logger.info(f"TF-IDF fitted: vocab={len(self._vocab)}, SVD dim={dim}")

    def _tfidf_matrix(self, texts: List[str], tokenised=None) -> np.ndarray:
        import re, math
        tokenise = lambda t: re.findall(r"[a-z]+", t.lower())
        if tokenised is None:
            tokenised = [tokenise(t) for t in texts]
        V = len(self._vocab)
        M = np.zeros((len(texts), V), dtype=np.float32)
        for i, tokens in enumerate(tokenised):
            tf = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            n = max(len(tokens), 1)
            for tok, cnt in tf.items():
                j = self._vocab.get(tok)
                if j is not None:
                    M[i, j] = (cnt / n) * self._idf[j]
        return M


    def encode_corpus(self, documents: List[Document]) -> np.ndarray:
        if self._backend == "dpr":
            return self._dpr_encode_corpus(documents)
        # TF-IDF path
        texts = [f"{d.title} {d.text}" for d in documents]
        self._fit_tfidf(texts)
        return self._tfidf_encode(texts)

    def encode_question(self, question: str) -> np.ndarray:
        if self._backend == "dpr":
            return self._dpr_encode_question(question)
        if self._vocab is None:
            raise RuntimeError("TF-IDF not fitted yet. Encode corpus first.")
        return self._tfidf_encode([question])[0]
