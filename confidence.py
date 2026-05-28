
import logging
import numpy as np
from typing import List, Tuple, Optional

from config import RAGConfig
from corpus import Document

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.method = cfg.confidence_method
        self.threshold = cfg.confidence_threshold


    def compute(
        self,
        retrieved: List[Tuple[Document, float]],
        all_scores: Optional[np.ndarray] = None,
    ) -> float:
        
        scores = np.array([s for _, s in retrieved], dtype=np.float32)
        if len(scores) == 0:
            return 0.0

        method = self.method
        if method == "gap":
            return self._gap(scores)
        elif method == "margin":
            return self._margin(scores)
        elif method == "entropy":
            return self._entropy(all_scores if all_scores is not None else scores)
        elif method == "combined":
            g = self._gap(scores)
            e = self._entropy(all_scores if all_scores is not None else scores)
            return 0.5 * g + 0.5 * e
        else:
            raise ValueError(f"Unknown confidence method: {method}")

    def should_abstain(self, confidence: float) -> bool:
        return confidence < self.threshold

    def confidence_tier(self, confidence: float) -> str:
        if confidence >= 0.5:
            return "HIGH"
        elif confidence >= self.threshold:
            return "MEDIUM"
        else:
            return "LOW (abstain)"


    @staticmethod
    def _gap(scores: np.ndarray) -> float:
        
        if len(scores) < 2:
            return 1.0
        s1, s2 = scores[0], scores[1]
        # Clip denominator to avoid division by zero
        denom = abs(s1) + 1e-9
        gap = (s1 - s2) / denom
        return float(np.clip(gap, 0.0, 1.0))

    @staticmethod
    def _margin(scores: np.ndarray) -> float:
        
    
        if len(scores) < 2:
            return 1.0
        margin = scores[0] - scores[1]
        return float(np.clip(margin / 2.0, 0.0, 1.0))   # normalise to [0,1]

    @staticmethod
    def _entropy(scores: np.ndarray) -> float:
        
        s = scores - scores.max()          
        probs = np.exp(s)
        probs /= probs.sum() + 1e-9
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        H = -np.sum(probs * np.log(probs))
        H_max = np.log(len(probs))
        if H_max < 1e-9:
            return 1.0
        return float(np.clip(1.0 - H / H_max, 0.0, 1.0))


    def explain(
        self,
        retrieved: List[Tuple[Document, float]],
        confidence: float,
    ) -> str:
        top = retrieved[:3]
        lines = [
            f"Confidence: {confidence:.3f} [{self.confidence_tier(confidence)}] "
            f"(method={self.method}, threshold={self.threshold})",
            f"Top-{len(top)} retrieved documents:",
        ]
        for rank, (doc, score) in enumerate(top, 1):
            lines.append(f"  {rank}. [{score:.4f}] {doc.title} (id={doc.id})")

        if len(retrieved) >= 2:
            gap = retrieved[0][1] - retrieved[1][1]
            lines.append(f"Score gap (rank1 - rank2): {gap:.4f}")

        if self.should_abstain(confidence):
            lines.append("Decision: ABSTAIN — confidence below threshold.")
        else:
            lines.append("Decision: GENERATE — confidence sufficient.")
        return "\n".join(lines)
