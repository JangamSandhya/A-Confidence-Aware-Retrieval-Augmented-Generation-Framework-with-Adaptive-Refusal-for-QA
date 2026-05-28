

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config import RAGConfig
from ragtruth_corpus import Document
from encoder import CorpusEncoder
from indexer import FAISSIndex
from confidence import ConfidenceEstimator
from generator import RAGGenerator

logger = logging.getLogger(__name__)



@dataclass
class RAGResult:
    question:       str
    answer:         str
    retrieved_docs: List[Tuple[Document, float]]
    confidence:     float
    decision:       str           
    mode:           str           
    explanation:    str = ""

    @property
    def abstained(self) -> bool:
        return self.decision == "ABSTAIN"

    @property
    def hedged(self) -> bool:
        return self.decision == "HEDGE"

    @property
    def top_doc_title(self) -> str:
        return self.retrieved_docs[0][0].title if self.retrieved_docs else "N/A"

    @property
    def top_doc_task(self) -> str:
        return self.retrieved_docs[0][0].task if self.retrieved_docs else "N/A"

    def to_dict(self) -> dict:
        return {
            "question":       self.question,
            "answer":         self.answer,
            "confidence":     round(self.confidence, 4),
            "decision":       self.decision,
            "abstained":      self.abstained,
            "hedged":         self.hedged,
            "mode":           self.mode,
            "top_doc_title":  self.top_doc_title,
            "top_doc_task":   self.top_doc_task,
            "explanation":    self.explanation,
        }




class ConfidenceAwareRAGPipeline:
   

    def __init__(
        self,
        encoder:    CorpusEncoder,
        index:      FAISSIndex,
        confidence: ConfidenceEstimator,
        generator:  RAGGenerator,
        cfg:        RAGConfig,
    ):
        self.encoder    = encoder
        self.index      = index
        self.confidence = confidence
        self.generator  = generator
        self.cfg        = cfg


    @classmethod
    def build(
        cls,
        cfg: RAGConfig,
        documents: Optional[List[Document]] = None,
        index_path: Optional[str] = None,
        docs_path:  Optional[str] = None,
    ) -> "ConfidenceAwareRAGPipeline":
        
        from ragtruth_corpus import RAGTruthCorpusLoader

        encoder    = CorpusEncoder(cfg)
        index      = FAISSIndex(cfg)
        confidence = ConfidenceEstimator(cfg)
        generator  = RAGGenerator(cfg)

        if documents is not None:
            logger.info(f"Encoding {len(documents)} documents with {encoder.backend} backend…")
            embs = encoder.encode_corpus(documents)
            index.build(embs, documents)
        elif index_path and docs_path:
            index.load(index_path, docs_path)
        else:
            # Load default corpus from config
            loader = RAGTruthCorpusLoader(cfg)
            docs   = loader.load()
            embs   = encoder.encode_corpus(docs)
            index.build(embs, docs)

        return cls(encoder=encoder, index=index,
                   confidence=confidence, generator=generator, cfg=cfg)


    def baseline_answer(self, question: str) -> RAGResult:
        retrieved, _ = self._retrieve(question)
        c = self.confidence.compute(retrieved)
        task = retrieved[0][0].task if retrieved else "qa"
        answer = self.generator.generate(question, retrieved,
                                         hedged=False, task=task)
        return RAGResult(
            question=question, answer=answer,
            retrieved_docs=retrieved, confidence=c,
            decision="GENERATE", mode="baseline",
        )

    def confidence_aware_answer(self, question: str) -> RAGResult:
        
        retrieved, all_scores = self._retrieve(question)
        c    = self.confidence.compute(retrieved, all_scores)
        dec  = self.confidence.decision(c)
        expl = self.confidence.explain(question, retrieved, c)
        task = retrieved[0][0].task if retrieved else "qa"

        if dec == "ABSTAIN":
            answer = self.cfg.abstention_message
        elif dec == "HEDGE":
            raw    = self.generator.generate(question, retrieved,
                                             hedged=True, task=task)
            answer = (f"[Retrieval confidence: {c:.2f} — moderate]\n{raw}")
        else:
            answer = self.generator.generate(question, retrieved,
                                             hedged=False, task=task)

        return RAGResult(
            question=question, answer=answer,
            retrieved_docs=retrieved, confidence=c,
            decision=dec, mode="confidence_aware",
            explanation=expl,
        )


    def _retrieve(
        self, question: str
    ) -> Tuple[List[Tuple[Document, float]], np.ndarray]:
        q_emb = self.encoder.encode_query(question)
        return self.index.search(q_emb, self.cfg.top_k)


    def interactive(self):
        print("\n" + "═" * 62)
        print("  RAGTruth Confidence-Aware RAG  —  Interactive Demo")
        print("  Commands:  'mode'  toggle mode   |   'quit'  exit")
        print("             'explain' show last confidence breakdown")
        print("═" * 62)
        mode   = "confidence_aware"
        last_result = None
        while True:
            try:
                q = input(f"\n[{mode}] ❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not q:
                continue
            if q.lower() == "quit":
                break
            if q.lower() == "mode":
                mode = "baseline" if mode == "confidence_aware" else "confidence_aware"
                print(f"  ↳ Switched to: {mode}")
                continue
            if q.lower() == "explain" and last_result:
                print(last_result.explanation)
                continue

            result = (self.confidence_aware_answer(q)
                      if mode == "confidence_aware"
                      else self.baseline_answer(q))
            last_result = result
            print(f"\n  Answer    : {result.answer[:300]}")
            print(f"  Confidence: {result.confidence:.4f}  "
                  f"[{self.confidence.tier(result.confidence)}]")
            print(f"  Decision  : {result.decision}")
            if result.retrieved_docs:
                t, s = result.retrieved_docs[0]
                print(f"  Top doc   : {t.title[:65]}  (score={s:.4f})")
