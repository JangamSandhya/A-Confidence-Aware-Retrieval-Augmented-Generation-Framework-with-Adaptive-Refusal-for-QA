

import json, logging, math, os, re, sys
import numpy as np
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
from ragtruth_corpus import RAGTRUTH_SAMPLE, Document
from evaluator import EVAL_QUESTIONS




class TFIDFEncoder:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self._vocab: Dict[str, int] = {}
        self._idf:   np.ndarray = np.array([])
        self._Vt:    np.ndarray = np.array([])

    def _tok(self, text: str) -> List[str]:
        return re.findall(r"[a-z]{2,}", text.lower())

    def fit(self, texts: List[str]) -> "TFIDFEncoder":
        n = len(texts)
        tok = [self._tok(t) for t in texts]
        df: Dict[str, int] = {}
        for t in tok:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        self._vocab = {w: i for i, w in enumerate(sorted(df))}
        self._idf   = np.array(
            [math.log((n+1)/(df[w]+1))+1.0 for w in sorted(df)], dtype=np.float32)
        M   = self._mat(tok)
        dim = min(self.dim, M.shape[0]-1, M.shape[1])
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        self._Vt = Vt[:dim]
        logger.info(f"TF-IDF fitted: vocab={len(self._vocab)}, dim={dim}, docs={n}")
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        tok = [self._tok(t) for t in texts]
        M   = self._mat(tok)
        R   = M @ self._Vt.T
        n   = np.linalg.norm(R, axis=1, keepdims=True) + 1e-9
        return (R / n).astype(np.float32)

    def _mat(self, tokenised: List[List[str]]) -> np.ndarray:
        V = len(self._vocab)
        M = np.zeros((len(tokenised), V), dtype=np.float32)
        for i, toks in enumerate(tokenised):
            tf: Dict[str, int] = {}
            for w in toks: tf[w] = tf.get(w, 0) + 1
            n = max(len(toks), 1)
            for w, c in tf.items():
                j = self._vocab.get(w)
                if j is not None:
                    M[i, j] = (c/n) * self._idf[j]
        return M




class NumpyRetriever:
    def __init__(self, enc: TFIDFEncoder):
        self.enc = enc
        self._embs: Optional[np.ndarray] = None
        self._docs: List[Document] = []

    def index(self, docs: List[Document]):
        self._docs = docs
        texts = [f"{d.question} {d.source_info}".strip() for d in docs]
        self.enc.fit(texts)
        self._embs = self.enc.encode(texts)

    def search(self, query: str, top_k: int = 5
               ) -> Tuple[List[Tuple[Document, float]], np.ndarray]:
        q = self.enc.encode([query])[0]
        scores = self._embs @ q
        idxs   = np.argsort(-scores)[:top_k]
        return [(self._docs[i], float(scores[i])) for i in idxs], scores



def conf_gap(scores: List[float]) -> float:
    if len(scores) < 2: return 1.0
    s1, s2 = scores[0], scores[1]
    return float(np.clip((s1-s2)/(abs(s1)+1e-9), 0, 1))

def conf_entropy(all_scores: np.ndarray) -> float:
    s = all_scores - all_scores.max()
    p = np.exp(s); p /= p.sum()+1e-9; p = np.clip(p, 1e-10, 1)
    H = -float(np.sum(p * np.log(p)))
    Hmax = math.log(len(p))
    return float(np.clip(1 - H/Hmax, 0, 1)) if Hmax > 0 else 1.0

def compute_confidence(top_scores: List[float], all_scores: np.ndarray,
                       method: str = "gap") -> float:
    if method == "gap":      return conf_gap(top_scores)
    if method == "margin":   return float(np.clip((top_scores[0]-top_scores[1])/2, 0, 1)) if len(top_scores)>=2 else 1.0
    if method == "entropy":  return conf_entropy(all_scores)
    if method == "combined": return 0.5*conf_gap(top_scores) + 0.5*conf_entropy(all_scores)
    raise ValueError(f"Unknown method: {method}")




def extractive_answer(question: str,
                      retrieved: List[Tuple[Document, float]]) -> str:
    q_toks = set(re.findall(r"[a-z]{3,}", question.lower()))
    best, best_score, best_title = "", -1, ""
    for doc, _ in retrieved[:3]:
        for sent in re.split(r"(?<=[.!?])\s+", doc.source_info):
            ov = len(q_toks & set(re.findall(r"[a-z]{3,}", sent.lower())))
            if ov > best_score:
                best_score, best, best_title = ov, sent.strip(), doc.title[:60]
    return f"{best}  (Source: {best_title})" if best else \
           f"{retrieved[0][0].source_info[:250]}…" if retrieved else "No context found."




def rouge1(hyp: str, ref: str) -> float:
    if not ref or ref == "N/A": return 0.0
    h = set(re.findall(r"[a-z]+", hyp.lower()))
    r = set(re.findall(r"[a-z]+", ref.lower()))
    ov = h & r
    if not ov: return 0.0
    p = len(ov)/len(h) if h else 0; rc = len(ov)/len(r) if r else 0
    return round(2*p*rc/(p+rc), 4) if (p+rc) > 0 else 0.0

def retrieval_support(answer: str, retrieved: List[Tuple[Document, float]]) -> bool:
    doc_text = " ".join(d.source_info for d, _ in retrieved).lower()
    stop = {"that","this","with","from","have","been","they","their","when","what",
            "which","will","more","also","note","based","answer","only","does",
            "information","moderate","confidence","retrieval","passage","context"}
    toks = set(re.findall(r"[a-z]{4,}", answer.lower())) - stop
    return bool(toks) and sum(1 for t in toks if t in doc_text)/len(toks) >= 0.3




def run_demo(
    threshold:  float = 0.15,
    high_tau:   float = 0.50,
    method:     str   = "gap",
    top_k:      int   = 5,
):
    ABSTENTION_MSG = (
        "The knowledge base does not contain sufficiently reliable information "
        "to answer this question confidently. Please consult a primary source."
    )

    logger.info("Loading RAGTruth corpus …")
    docs = [Document.from_dict(d) for d in RAGTRUTH_SAMPLE]
    enc  = TFIDFEncoder(dim=128)
    ret  = NumpyRetriever(enc)
    ret.index(docs)

    questions = EVAL_QUESTIONS
    logger.info(f"Evaluating {len(questions)} questions  "
                f"(threshold={threshold}, method={method}) …")

    baseline_results, ca_results = [], []

    for q_data in questions:
        q   = q_data["question"]
        ref = q_data.get("reference", "N/A")
        dom = q_data.get("domain", "in")

        retrieved, all_scores = ret.search(q, top_k)
        top_scores = [s for _, s in retrieved]
        conf = compute_confidence(top_scores, all_scores, method)

        b_ans = extractive_answer(q, retrieved)
        baseline_results.append({
            "id": q_data["id"], "question": q, "answer": b_ans,
            "confidence": round(conf, 4), "decision": "GENERATE",
            "abstained": False, "hedged": False,
            "rouge1": rouge1(b_ans, ref),
            "supported": retrieval_support(b_ans, retrieved),
            "reference": ref, "domain": dom,
            "top_doc": retrieved[0][0].title if retrieved else "N/A",
        })

        if conf < threshold:
            decision = "ABSTAIN"
            ca_ans   = ABSTENTION_MSG
        elif conf < high_tau:
            decision = "HEDGE"
            ca_ans   = (f"[Retrieval confidence: {conf:.2f} — moderate]\n"
                        + extractive_answer(q, retrieved))
        else:
            decision = "GENERATE"
            ca_ans   = extractive_answer(q, retrieved)

        ca_results.append({
            "id": q_data["id"], "question": q, "answer": ca_ans,
            "confidence": round(conf, 4), "decision": decision,
            "abstained": decision == "ABSTAIN",
            "hedged":    decision == "HEDGE",
            "rouge1": rouge1(ca_ans, ref) if decision != "ABSTAIN" else 0.0,
            "supported": (retrieval_support(ca_ans, retrieved)
                          if decision != "ABSTAIN" else None),
            "reference": ref, "domain": dom,
            "top_doc": retrieved[0][0].title if retrieved else "N/A",
        })

    _print_summary(baseline_results, ca_results, threshold, high_tau, method)

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_results.json",         "w") as f: json.dump(baseline_results, f, indent=2)
    with open("results/confidence_aware_results.json", "w") as f: json.dump(ca_results,       f, indent=2)
    _write_csv(baseline_results, ca_results)
    logger.info("Results saved to results/")

    return baseline_results, ca_results




def _print_summary(baseline, ca, tau, tau_h, method):
    n   = len(baseline)
    sep = "═" * 68

    def stats(results):
        abstain = sum(1 for r in results if r.get("abstained"))
        hedged  = sum(1 for r in results if r.get("hedged"))
        ans     = n - abstain
        rg      = sum(r["rouge1"] for r in results if not r.get("abstained")) / max(ans,1)
        sup_l   = [r for r in results if r.get("supported") is not None]
        sup     = sum(1 for r in sup_l if r["supported"]) / max(len(sup_l),1)
        conf    = sum(r["confidence"] for r in results) / n
        ood     = [r for r in results if r.get("domain") == "ood"]
        ood_abs = sum(1 for r in ood if r.get("abstained"))
        return abstain, hedged, ans, rg, sup, conf, ood_abs, len(ood)

    b = stats(baseline)
    c = stats(ca)

    print(f"\n{sep}")
    print("  EVALUATION SUMMARY  —  RAGTruth Confidence-Aware RAG Demo")
    print(sep)
    print(f"  Total questions   : {n}")
    print(f"  Corpus            : RAGTruth ({len(RAGTRUTH_SAMPLE)} documents, 3 tasks)")
    print(f"  Confidence method : {method}  |  τ={tau}  τ_H={tau_h}")
    print()
    print("  BASELINE RAG:")
    print(f"    Avg ROUGE-1 F1    : {b[3]:.4f}")
    print(f"    Retrieval support : {b[4]:.2%}")
    print(f"    Avg confidence    : {b[5]:.4f}")
    print(f"    Abstention rate   : 0.00%  (never abstains)")
    print(f"    OOD abstention    : {b[6]}/{b[7]}  (0%)")
    print()
    print("  CONFIDENCE-AWARE RAG:")
    print(f"    Avg ROUGE-1 F1    : {c[3]:.4f}  (answered qs only)")
    print(f"    Retrieval support : {c[4]:.2%}  (answered qs only)")
    print(f"    Avg confidence    : {c[5]:.4f}")
    print(f"    Abstention rate   : {c[0]/n:.2%}  ({c[0]}/{n})")
    print(f"    Hedged rate       : {c[1]/n:.2%}  ({c[1]}/{n})")
    print(f"    Answer coverage   : {c[2]/n:.2%}  ({c[2]}/{n})")
    print(f"    OOD abstention    : {c[6]}/{c[7]}  ({c[6]/max(c[7],1):.0%})")
    print()
    print("  IMPROVEMENT (answered questions):")
    print(f"    ROUGE-1 delta     : {c[3]-b[3]:+.4f}")
    print(f"    Support rate Δ    : {c[4]-b[4]:+.2%}")
    print(sep)

    print("\n  SAMPLE (first 10 questions):\n")
    print(f"  {'ID':<5} {'Conf':>6} {'Dec':<9} {'Dom':<4} {'B-R1':>6} {'CA-R1':>6}  Question")
    print("  " + "─" * 70)
    for b_r, c_r in list(zip(baseline, ca))[:10]:
        dec_s = {"ABSTAIN":"ABS","HEDGE":"HED","GENERATE":"GEN"}.get(
            c_r["decision"], c_r["decision"][:3])
        print(f"  {b_r['id']:<5} {b_r['confidence']:>6.3f} {dec_s:<9} "
              f"{b_r.get('domain','?'):<4} {b_r['rouge1']:>6.4f} {c_r['rouge1']:>6.4f}  "
              f"{b_r['question'][:40]}")
    print()


def _write_csv(baseline, ca):
    import csv
    path = "results/comparison.csv"
    fields = ["id","domain","question","confidence",
              "b_rouge1","ca_rouge1","ca_decision",
              "b_supported","ca_supported","b_answer","ca_answer","reference"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for b, c in zip(baseline, ca):
            w.writerow({
                "id": b["id"], "domain": b.get("domain",""),
                "question": b["question"], "confidence": b["confidence"],
                "b_rouge1": b["rouge1"], "ca_rouge1": c["rouge1"],
                "ca_decision": c["decision"],
                "b_supported": b.get("supported",""),
                "ca_supported": c.get("supported",""),
                "b_answer": b["answer"][:120], "ca_answer": c["answer"][:120],
                "reference": b.get("reference",""),
            })
    logger.info(f"CSV → {path}")



if __name__ == "__main__":
    run_demo()
