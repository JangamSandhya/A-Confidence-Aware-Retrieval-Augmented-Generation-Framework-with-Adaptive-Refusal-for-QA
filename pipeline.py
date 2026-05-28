

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from config import RAGConfig
from corpus import Document
from retriever import DPRRetriever
from indexer import FAISSIndexer
from generator import RAGGenerator
from confidence import ConfidenceEstimator

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieved_docs: List[Tuple[Document, float]]
    confidence: float
    abstained: bool
    hedged: bool
    mode: str           
    explanation: str = ""

    @property
    def top_doc_title(self) -> str:
        if self.retrieved_docs:
            return self.retrieved_docs[0][0].title
        return "N/A"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "abstained": self.abstained,
            "hedged": self.hedged,
            "mode": self.mode,
            "top_doc": self.top_doc_title,
            "explanation": self.explanation,
        }


class ConfidenceAwareRAGPipeline:
   

    HIGH_CONFIDENCE_THRESHOLD = 0.5   

    def __init__(
        self,
        retriever: DPRRetriever,
        indexer: FAISSIndexer,
        generator: RAGGenerator,
        confidence: ConfidenceEstimator,
        cfg: RAGConfig,
    ):
        self.retriever = retriever
        self.indexer = indexer
        self.generator = generator
        self.confidence = confidence
        self.cfg = cfg


    def baseline_answer(self, question: str) -> RAGResult:
        retrieved, _ = self._retrieve(question)
        conf = self.confidence.compute(retrieved)
        answer = self.generator.generate(question, retrieved, hedged=False)
        return RAGResult(
            question=question,
            answer=answer,
            retrieved_docs=retrieved,
            confidence=conf,
            abstained=False,
            hedged=False,
            mode="baseline",
        )

    def confidence_aware_answer(self, question: str) -> RAGResult:
        
        
        retrieved, all_scores = self._retrieve(question)
        conf = self.confidence.compute(retrieved, all_scores)
        explanation = self.confidence.explain(retrieved, conf)

        abstained = False
        hedged = False

        if self.confidence.should_abstain(conf):
            abstained = True
            answer = self.cfg.abstention_message
        elif conf < self.HIGH_CONFIDENCE_THRESHOLD:
            hedged = True
            answer = self.generator.generate(question, retrieved, hedged=True)
            answer = f"[Note: Moderate retrieval confidence ({conf:.2f})]\n{answer}"
        else:
            answer = self.generator.generate(question, retrieved, hedged=False)

        return RAGResult(
            question=question,
            answer=answer,
            retrieved_docs=retrieved,
            confidence=conf,
            abstained=abstained,
            hedged=hedged,
            mode="confidence_aware",
            explanation=explanation,
        )


    def _retrieve(
        self, question: str
    ) -> Tuple[List[Tuple[Document, float]], np.ndarray]:
        q_emb = self.retriever.encode_question(question)
        retrieved, all_scores = self.indexer.search_with_all_scores(
            q_emb, self.cfg.top_k
        )
        return retrieved, all_scores


    def interactive(self):
        print("\n" + "=" * 60)
        print("  Confidence-Aware RAG — Interactive Demo")
        print("  Type 'quit' to exit, 'mode' to toggle mode.")
        print("=" * 60)
        mode = "confidence_aware"
        while True:
            try:
                q = input(f"\n[{mode}] Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if q.lower() == "quit":
                break
            if q.lower() == "mode":
                mode = "baseline" if mode == "confidence_aware" else "confidence_aware"
                print(f"  ↳ Mode switched to: {mode}")
                continue
            if not q:
                continue

            result = (
                self.confidence_aware_answer(q)
                if mode == "confidence_aware"
                else self.baseline_answer(q)
            )
            print(f"\n  Answer: {result.answer}")
            print(f"  Confidence: {result.confidence:.3f} | "
                  f"Abstained: {result.abstained} | Hedged: {result.hedged}")
            if result.retrieved_docs:
                print(f"  Top doc: {result.retrieved_docs[0][0].title} "
                      f"(score={result.retrieved_docs[0][1]:.4f})")
