

import json
import logging
import os
import sys
import re
import math
import numpy as np
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

from corpus import PUBMED_SAMPLE, Document
from evaluator import DEFAULT_EVAL_QUESTIONS



class TFIDFEncoder:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.array([])
        self._svd_Vt: np.ndarray = np.array([])

    def _tokenise(self, text: str) -> List[str]:
        return re.findall(r"[a-z]{2,}", text.lower())

    def fit(self, texts: List[str]):
        n = len(texts)
        tokenised = [self._tokenise(t) for t in texts]
        df: Dict[str, int] = {}
        for toks in tokenised:
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1
        self._vocab = {w: i for i, w in enumerate(sorted(df))}
        self._idf = np.array(
            [math.log((n + 1) / (df[w] + 1)) + 1.0 for w in sorted(df)],
            dtype=np.float32,
        )
        M = self._tfidf_matrix(tokenised)
        dim = min(self.dim, M.shape[1])
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        self._svd_Vt = Vt[:dim]

    def encode(self, texts: List[str]) -> np.ndarray:
        tokenised = [self._tokenise(t) for t in texts]
        M = self._tfidf_matrix(tokenised)
        reduced = M @ self._svd_Vt.T
        norms = np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-9
        return (reduced / norms).astype(np.float32)

    def _tfidf_matrix(self, tokenised: List[List[str]]) -> np.ndarray:
        V = len(self._vocab)
        M = np.zeros((len(tokenised), V), dtype=np.float32)
        for i, toks in enumerate(tokenised):
            tf: Dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            n = max(len(toks), 1)
            for tok, cnt in tf.items():
                j = self._vocab.get(tok)
                if j is not None:
                    M[i, j] = (cnt / n) * self._idf[j]
        return M



class NumpyRetriever:
    def __init__(self, encoder: TFIDFEncoder):
        self.encoder = encoder
        self._corpus_emb: Optional[np.ndarray] = None
        self._docs: List[Document] = []

    def index(self, docs: List[Document]):
        self._docs = docs
        texts = [f"{d.title} {d.text}" for d in docs]
        self.encoder.fit(texts)
        self._corpus_emb = self.encoder.encode(texts)
        logger.info(f"Indexed {len(docs)} documents (dim={self._corpus_emb.shape[1]})")

    def search(self, query: str, top_k: int) -> Tuple[List[Tuple[Document, float]], np.ndarray]:
        q_emb = self.encoder.encode([query])[0]
        scores = self._corpus_emb @ q_emb
        idxs = np.argsort(-scores)[:top_k]
        results = [(self._docs[i], float(scores[i])) for i in idxs]
        return results, scores



def extractive_generate(question: str, retrieved: List[Tuple[Document, float]]) -> str:
    q_tokens = set(re.findall(r"[a-z]{3,}", question.lower()))
    best_sent, best_score = None, -1
    for doc, _ in retrieved[:3]:
        sents = re.split(r"(?<=[.!?])\s+", doc.text)
        for sent in sents:
            overlap = len(q_tokens & set(re.findall(r"[a-z]{3,}", sent.lower())))
            if overlap > best_score:
                best_score = overlap
                best_sent = (sent, doc.title)
    if best_sent:
        return f"{best_sent[0]}  (Source: {best_sent[1]})"
    if retrieved:
        d = retrieved[0][0]
        return f"{d.text[:200]}…  (Source: {d.title})"
    return "No relevant information found."



def compute_confidence(scores_top: List[float], all_scores: np.ndarray,
                       method: str = "gap") -> float:
    if not scores_top:
        return 0.0
    if method == "gap":
        if len(scores_top) < 2:
            return 1.0
        s1, s2 = scores_top[0], scores_top[1]
        gap = (s1 - s2) / (abs(s1) + 1e-9)
        return float(np.clip(gap, 0.0, 1.0))
    elif method == "entropy":
        s = all_scores - all_scores.max()
        p = np.exp(s); p /= p.sum() + 1e-9
        p = np.clip(p, 1e-10, 1.0)
        H = -np.sum(p * np.log(p))
        H_max = math.log(len(p))
        return float(np.clip(1.0 - H / H_max, 0.0, 1.0)) if H_max > 0 else 1.0
    else:
        g = compute_confidence(scores_top, all_scores, "gap")
        e = compute_confidence(scores_top, all_scores, "entropy")
        return 0.5 * g + 0.5 * e



def run_demo():
    THRESHOLD = 0.15
    TOP_K = 5
    METHOD = "gap"
    ABSTENTION_MSG = (
        "I could not find sufficiently reliable information in the knowledge "
        "base to answer this question confidently. Please consult a domain expert."
    )

    logger.info("Loading corpus…")
    docs = [Document(d["id"], d["title"], d["text"]) for d in PUBMED_SAMPLE]

    logger.info("Building TF-IDF index…")
    encoder = TFIDFEncoder(dim=128)
    retriever = NumpyRetriever(encoder)
    retriever.index(docs)

    logger.info("Running evaluation on 34 questions…")
    questions = DEFAULT_EVAL_QUESTIONS

    baseline_results = []
    ca_results = []

    for q_data in questions:
        q = q_data["question"]
        ref = q_data.get("reference", "N/A")

        retrieved, all_scores = retriever.search(q, TOP_K)
        top_scores = [s for _, s in retrieved]
        conf = compute_confidence(top_scores, all_scores, METHOD)

        b_answer = extractive_generate(q, retrieved)
        b_rouge = rouge1(b_answer, ref)
        b_support = has_retrieval_support(b_answer, retrieved)

        baseline_results.append({
            "id": q_data["id"],
            "question": q,
            "answer": b_answer,
            "confidence": round(conf, 4),
            "abstained": False,
            "hedged": False,
            "rouge1": b_rouge,
            "has_retrieval_support": b_support,
            "reference": ref,
        })

        abstained = False
        hedged = False
        if conf < THRESHOLD:
            ca_answer = ABSTENTION_MSG
            abstained = True
        elif conf < 0.5:
            ca_answer = f"[Moderate confidence: {conf:.2f}] " + extractive_generate(q, retrieved)
            hedged = True
        else:
            ca_answer = extractive_generate(q, retrieved)

        ca_rouge = rouge1(ca_answer, ref) if not abstained else 0.0
        ca_support = has_retrieval_support(ca_answer, retrieved) if not abstained else None

        ca_results.append({
            "id": q_data["id"],
            "question": q,
            "answer": ca_answer,
            "confidence": round(conf, 4),
            "abstained": abstained,
            "hedged": hedged,
            "rouge1": ca_rouge,
            "has_retrieval_support": ca_support,
            "reference": ref,
        })

    print_summary(baseline_results, ca_results, THRESHOLD)

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    with open("results/confidence_aware_results.json", "w") as f:
        json.dump(ca_results, f, indent=2)
    write_comparison_csv(baseline_results, ca_results)

    logger.info("Demo complete. Results saved to results/")
    return baseline_results, ca_results



def rouge1(hypothesis: str, reference: str) -> float:
    if not reference or reference == "N/A":
        return 0.0
    h = set(re.findall(r"[a-z]+", hypothesis.lower()))
    r = set(re.findall(r"[a-z]+", reference.lower()))
    overlap = h & r
    if not overlap:
        return 0.0
    p = len(overlap) / len(h) if h else 0.0
    rc = len(overlap) / len(r) if r else 0.0
    return round(2 * p * rc / (p + rc), 4) if (p + rc) > 0 else 0.0


def has_retrieval_support(answer: str, retrieved) -> bool:
    doc_text = " ".join(d.text for d, _ in retrieved).lower()
    tokens = set(re.findall(r"[a-z]{4,}", answer.lower()))
    stop = {"that", "this", "with", "from", "have", "been", "they", "their",
            "when", "what", "which", "will", "more", "also", "note", "based",
            "answer", "only", "does", "information", "moderate", "confidence"}
    tokens -= stop
    if not tokens:
        return False
    return sum(1 for t in tokens if t in doc_text) / len(tokens) >= 0.3


def print_summary(baseline: List[Dict], ca: List[Dict], threshold: float):
    n = len(baseline)
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY — Confidence-Aware RAG Demo")
    print("=" * 70)
    print(f"  Total questions : {n}")
    print(f"  Conf threshold  : {threshold}")
    print()

    b_rouge = sum(r["rouge1"] for r in baseline) / n
    b_sup = sum(1 for r in baseline if r.get("has_retrieval_support")) / n
    b_conf = sum(r["confidence"] for r in baseline) / n
    print("  BASELINE RAG:")
    print(f"    Avg ROUGE-1 F1    : {b_rouge:.4f}")
    print(f"    Retrieval support : {b_sup:.2%}")
    print(f"    Avg confidence    : {b_conf:.4f}")
    print(f"    Abstention rate   : 0.00%")
    print()

    abstained = sum(1 for r in ca if r["abstained"])
    hedged = sum(1 for r in ca if r.get("hedged"))
    answered = n - abstained
    ca_rouge = sum(r["rouge1"] for r in ca if not r["abstained"]) / max(answered, 1)
    sup_list = [r for r in ca if r.get("has_retrieval_support") is not None]
    ca_sup = sum(1 for r in sup_list if r["has_retrieval_support"]) / max(len(sup_list), 1)
    ca_conf = sum(r["confidence"] for r in ca) / n
    print("  CONFIDENCE-AWARE RAG:")
    print(f"    Avg ROUGE-1 F1    : {ca_rouge:.4f}  (answered qs only)")
    print(f"    Retrieval support : {ca_sup:.2%}  (answered qs only)")
    print(f"    Avg confidence    : {ca_conf:.4f}")
    print(f"    Abstention rate   : {abstained/n:.2%}  ({abstained}/{n})")
    print(f"    Hedged rate       : {hedged/n:.2%}  ({hedged}/{n})")
    print(f"    Answer coverage   : {answered/n:.2%}  ({answered}/{n})")
    print()
    print(f"  IMPROVEMENT (answered qs):")
    print(f"    ROUGE-1 delta     : {ca_rouge - b_rouge:+.4f}")
    print(f"    Support rate Δ    : {ca_sup - b_sup:+.2%}")
    print("=" * 70)

    print("\n  SAMPLE RESULTS (first 10 questions):\n")
    header = f"  {'ID':<5} {'Conf':>6} {'Abs':>4} {'B-R1':>6} {'CA-R1':>6}  Question"
    print(header)
    print("  " + "-" * 68)
    for b, c in list(zip(baseline, ca))[:10]:
        print(
            f"  {b['id']:<5} {b['confidence']:>6.3f} {'Y' if c['abstained'] else 'N':>4} "
            f"{b['rouge1']:>6.4f} {c['rouge1']:>6.4f}  {b['question'][:45]}"
        )
    print()


def write_comparison_csv(baseline: List[Dict], ca: List[Dict]):
    import csv
    path = "results/comparison.csv"
    fields = ["id", "question", "confidence", "b_rouge1", "ca_rouge1",
              "ca_abstained", "ca_hedged", "b_support", "ca_support",
              "b_answer_snippet", "ca_answer_snippet", "reference"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for b, c in zip(baseline, ca):
            w.writerow({
                "id": b["id"],
                "question": b["question"],
                "confidence": b["confidence"],
                "b_rouge1": b["rouge1"],
                "ca_rouge1": c["rouge1"],
                "ca_abstained": c["abstained"],
                "ca_hedged": c.get("hedged", False),
                "b_support": b.get("has_retrieval_support", ""),
                "ca_support": c.get("has_retrieval_support", ""),
                "b_answer_snippet": b["answer"][:100],
                "ca_answer_snippet": c["answer"][:100],
                "reference": b.get("reference", ""),
            })
    logger.info(f"CSV saved → {path}")


if __name__ == "__main__":
    run_demo()
