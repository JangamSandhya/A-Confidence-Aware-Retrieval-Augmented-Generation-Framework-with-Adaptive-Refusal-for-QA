

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from config import RAGConfig
from ragtruth_corpus import Document

logger = logging.getLogger(__name__)


class ConfidenceEstimator:

    TIER_HIGH   = "HIGH"
    TIER_MEDIUM = "MEDIUM"
    TIER_LOW    = "LOW (abstain)"

    def __init__(self, cfg: RAGConfig):
        self.cfg       = cfg
        self.method    = cfg.confidence_method
        self.tau       = cfg.confidence_threshold
        self.tau_high  = cfg.high_conf_threshold


    def compute(
        self,
        retrieved: List[Tuple[Document, float]],
        all_scores: Optional[np.ndarray] = None,
    ) -> float:
        
        scores = np.array([s for _, s in retrieved], dtype=np.float32)
        if len(scores) == 0:
            return 0.0

        if   self.method == "gap":
            return self._gap(scores)
        elif self.method == "margin":
            return self._margin(scores)
        elif self.method == "entropy":
            src = all_scores if all_scores is not None else scores
            return self._entropy(src)
        elif self.method == "combined":
            g = self._gap(scores)
            src = all_scores if all_scores is not None else scores
            e = self._entropy(src)
            return float(0.5 * g + 0.5 * e)
        else:
            raise ValueError(f"Unknown confidence method: {self.method!r}")

    def decision(self, c: float) -> str:

        if c < self.tau:
            return "ABSTAIN"
        elif c < self.tau_high:
            return "HEDGE"
        return "GENERATE"

    def should_abstain(self, c: float) -> bool:
        return c < self.tau

    def tier(self, c: float) -> str:
        if c >= self.tau_high:
            return self.TIER_HIGH
        elif c >= self.tau:
            return self.TIER_MEDIUM
        return self.TIER_LOW


    @staticmethod
    def _gap(scores: np.ndarray) -> float:
        
        if len(scores) < 2:
            return 1.0
        s1, s2 = float(scores[0]), float(scores[1])
        gap = (s1 - s2) / (abs(s1) + 1e-9)
        return float(np.clip(gap, 0.0, 1.0))

    @staticmethod
    def _margin(scores: np.ndarray) -> float:
        
        if len(scores) < 2:
            return 1.0
        margin = (float(scores[0]) - float(scores[1])) / 2.0
        return float(np.clip(margin, 0.0, 1.0))

    @staticmethod
    def _entropy(scores: np.ndarray) -> float:
        
        s = scores - scores.max()          
        p = np.exp(s)
        p /= p.sum() + 1e-9
        p = np.clip(p, 1e-10, 1.0)
        H     = -float(np.sum(p * np.log(p)))
        H_max = math.log(len(p))
        if H_max < 1e-9:
            return 1.0
        return float(np.clip(1.0 - H / H_max, 0.0, 1.0))


    def explain(
        self,
        query: str,
        retrieved: List[Tuple[Document, float]],
        c: float,
    ) -> str:
        lines = [
            f"Query      : {query}",
            f"Confidence : {c:.4f}  [{self.tier(c)}]",
            f"Method     : {self.method}  |  τ={self.tau}  τ_H={self.tau_high}",
            f"Decision   : {self.decision(c)}",
            "─" * 55,
            f"Top-{min(len(retrieved), 3)} retrieved documents:",
        ]
        for rank, (doc, score) in enumerate(retrieved[:3], 1):
            lines.append(f"  {rank}. [{score:.4f}]  {doc.title[:70]}")
        if len(retrieved) >= 2:
            g = retrieved[0][1] - retrieved[1][1]
            lines.append(f"Score gap (rank-1 − rank-2) : {g:.4f}")
        return "\n".join(lines)
