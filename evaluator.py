

import csv
import json
import logging
import os
import re
from typing import Dict, List, Optional

from config import RAGConfig
from pipeline import ConfidenceAwareRAGPipeline, RAGResult

logger = logging.getLogger(__name__)




EVAL_QUESTIONS: List[Dict] = [
    # RAG / Retrieval 
    {"id": "q01", "question": "What is retrieval-augmented generation?",
     "reference": "RAG combines a retrieval component with a generative language model by retrieving relevant documents and providing them as context for generation.", "domain": "in"},
    {"id": "q02", "question": "What is hallucination in language models?",
     "reference": "Hallucination is when a model generates text that is factually incorrect, inconsistent with the source, or fabricated without grounding in verifiable evidence.", "domain": "in"},
    {"id": "q03", "question": "What is Dense Passage Retrieval?",
     "reference": "DPR is a bi-encoder retrieval framework using two BERT-based encoders trained with contrastive objectives to maximise similarity between questions and relevant passages.", "domain": "in"},
    {"id": "q04", "question": "How does FAISS enable fast similarity search?",
     "reference": "FAISS supports exact and approximate nearest-neighbour search using flat indices, inverted file indices, and HNSW graph-based search with optional GPU acceleration.", "domain": "in"},
    {"id": "q05", "question": "What is the transformer architecture?",
     "reference": "The transformer uses self-attention mechanisms with multi-head attention and feed-forward sub-layers with residual connections, enabling parallel processing of sequences.", "domain": "in"},
    {"id": "q06", "question": "How is BERT pre-trained?",
     "reference": "BERT is pre-trained on masked language modelling and next sentence prediction using a bidirectional transformer encoder.", "domain": "in"},
    {"id": "q07", "question": "How is TF-IDF computed?",
     "reference": "TF-IDF multiplies term frequency by inverse document frequency, giving high weight to terms frequent in a document but rare across the corpus.", "domain": "in"},
    {"id": "q08", "question": "What is the difference between open-domain and closed-domain QA?",
     "reference": "Open-domain QA uses large unrestricted knowledge sources while closed-domain QA restricts retrieval to a specific subject area for higher precision.", "domain": "in"},
    {"id": "q09", "question": "What is BM25?",
     "reference": "BM25 is a probabilistic sparse retrieval algorithm extending TF-IDF with document length normalisation and term frequency saturation.", "domain": "in"},
    {"id": "q10", "question": "What is the Fusion-in-Decoder approach?",
     "reference": "FiD encodes each retrieved passage independently and fuses all encoded representations in the decoder, enabling evidence aggregation across multiple passages.", "domain": "in"},
    # Self-RAG / Confidence
    {"id": "q11", "question": "What is Self-RAG?",
     "reference": "Self-RAG trains a model to generate reflection tokens that decide whether to retrieve, assess passage relevance, and evaluate output support.", "domain": "in"},
    {"id": "q12", "question": "What is entropy in information theory?",
     "reference": "Shannon entropy measures the uncertainty of a probability distribution as the negative sum of p log p, maximised by uniform distributions.", "domain": "in"},
    {"id": "q13", "question": "How does GPT differ from BERT?",
     "reference": "GPT is an autoregressive decoder using left-to-right masked attention for generation, while BERT uses bidirectional attention for understanding tasks.", "domain": "in"},
    {"id": "q14", "question": "What are vector databases used for in RAG?",
     "reference": "Vector databases index high-dimensional embeddings and support approximate nearest-neighbour search to retrieve semantically similar passages for a query.", "domain": "in"},
    {"id": "q15", "question": "What is contrastive learning?",
     "reference": "Contrastive learning trains encoders by bringing positive pair representations closer and pushing negative pair representations apart in the embedding space.", "domain": "in"},
    # Summarisation / Data-to-Text 
    {"id": "q16", "question": "What is chain-of-thought prompting?",
     "reference": "Chain-of-thought prompting encourages models to produce intermediate reasoning steps before the final answer, improving performance on complex reasoning tasks.", "domain": "in"},
    {"id": "q17", "question": "What is instruction tuning?",
     "reference": "Instruction tuning fine-tunes language models on diverse tasks described using natural language instructions to improve zero-shot performance.", "domain": "in"},
    {"id": "q18", "question": "What is prompt engineering?",
     "reference": "Prompt engineering involves designing input prompts with techniques like few-shot examples, chain-of-thought, and role prompting to elicit desired model behaviour.", "domain": "in"},
    {"id": "q19", "question": "What are knowledge graphs?",
     "reference": "Knowledge graphs represent factual knowledge as entity-relation-entity triples and can provide structured grounding for RAG systems alongside unstructured retrieval.", "domain": "in"},
    {"id": "q20", "question": "What are ROUGE metrics?",
     "reference": "ROUGE metrics measure n-gram overlap between generated and reference text; ROUGE-1 uses unigrams, ROUGE-2 bigrams, and ROUGE-L longest common subsequence.", "domain": "in"},
    # Hallucination / RAGTruth 
    {"id": "q21", "question": "What makes a RAG system hallucinate?",
     "reference": "RAG hallucination occurs from retrieval failure, grounding failure where the generator ignores context, or aggregation failure combining passages incorrectly.", "domain": "in"},
    {"id": "q22", "question": "How does selective prediction help in QA?",
     "reference": "Selective prediction allows systems to abstain when uncertain, trading coverage for accuracy by refusing questions most likely to be answered incorrectly.", "domain": "in"},
    {"id": "q23", "question": "What is domain adaptation in NLP?",
     "reference": "Domain adaptation adapts models trained on general data to a specific target domain through fine-tuning, continued pre-training, or retrieval-based adaptation.", "domain": "in"},
    {"id": "q24", "question": "What is the RAGTruth benchmark?",
     "reference": "RAGTruth is a hallucination benchmark with human-annotated span-level labels across QA, summarisation, and data-to-text tasks for RAG systems.", "domain": "in"},
    {"id": "q25", "question": "How is cosine similarity computed?",
     "reference": "Cosine similarity is the dot product of two vectors divided by the product of their L2 norms, measuring directional similarity in the embedding space.", "domain": "in"},
    # Edge cases 
    {"id": "q26", "question": "What is beam search?",
     "reference": "Beam search maintains a fixed number of candidate sequences at each decoding step, keeping only the top-k by cumulative log-probability.", "domain": "in"},
    {"id": "q27", "question": "What is singular value decomposition?",
     "reference": "SVD decomposes a matrix into U Σ V^T where truncated SVD retains the top-k singular values for dimensionality reduction.", "domain": "in"},
    {"id": "q28", "question": "What is the difference between precision and recall?",
     "reference": "Precision is TP/(TP+FP) measuring retrieved positives that are correct; recall is TP/(TP+FN) measuring correct positives that are retrieved.", "domain": "in"},
    {"id": "q29", "question": "What determines large language model capability?",
     "reference": "LLM capability is primarily determined by parameter count, training data scale, and instruction tuning quality.", "domain": "in"},
    {"id": "q30", "question": "What DPR models are used for question encoding?",
     "reference": "DPR uses facebook/dpr-question_encoder-single-nq-base for questions and facebook/dpr-ctx_encoder-single-nq-base for passages.", "domain": "in"},
    # Out-of-domain 
    {"id": "q31", "question": "What is the capital city of Australia?",
     "reference": "N/A", "domain": "ood"},
    {"id": "q32", "question": "Who composed the opera La Traviata?",
     "reference": "N/A", "domain": "ood"},
    {"id": "q33", "question": "What is the chemical formula for glucose?",
     "reference": "N/A", "domain": "ood"},
    {"id": "q34", "question": "How many planets are in the solar system?",
     "reference": "N/A", "domain": "ood"},
]



class RAGEvaluator:

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg


    def load_questions(self) -> List[Dict]:
        path = self.cfg.eval_questions_path
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        logger.info("Using built-in RAGTruth-aligned evaluation set (34 questions).")
        return EVAL_QUESTIONS


    def evaluate(
        self,
        pipeline: ConfidenceAwareRAGPipeline,
        questions: List[Dict],
        mode: str = "baseline",
    ) -> List[Dict]:
        results = []
        for i, q_data in enumerate(questions, 1):
            q   = q_data["question"]
            ref = q_data.get("reference", "N/A")
            dom = q_data.get("domain", "in")
            logger.info(f"  [{mode}] Q{i:02d} ({dom}): {q[:55]}…")

            result: RAGResult = (
                pipeline.baseline_answer(q)
                if mode == "baseline"
                else pipeline.confidence_aware_answer(q)
            )

            entry = result.to_dict()
            entry.update({
                "id":        q_data.get("id", str(i)),
                "reference": ref,
                "domain":    dom,
                "rouge1":    self._rouge1(result.answer, ref) if not result.abstained else 0.0,
                "supported": (self._retrieval_support(result.answer, result.retrieved_docs)
                              if not result.abstained else None),
            })
            results.append(entry)
        return results


    @staticmethod
    def _rouge1(hypothesis: str, reference: str) -> float:
        if not reference or reference == "N/A":
            return 0.0
        h = set(re.findall(r"[a-z]+", hypothesis.lower()))
        r = set(re.findall(r"[a-z]+", reference.lower()))
        overlap = h & r
        if not overlap:
            return 0.0
        p  = len(overlap) / len(h) if h else 0.0
        rc = len(overlap) / len(r) if r else 0.0
        return round(2 * p * rc / (p + rc), 4) if (p + rc) > 0 else 0.0

    @staticmethod
    def _retrieval_support(answer: str, retrieved_docs) -> bool:
        if not retrieved_docs:
            return False
        doc_text = " ".join(d.source_info for d, _ in retrieved_docs).lower()
        stop = {"that", "this", "with", "from", "have", "been", "they", "their",
                "when", "what", "which", "will", "more", "also", "note", "based",
                "answer", "only", "does", "information", "moderate", "confidence",
                "retrieval", "passage", "context", "source", "provided"}
        tokens = set(re.findall(r"[a-z]{4,}", answer.lower())) - stop
        if not tokens:
            return False
        return sum(1 for t in tokens if t in doc_text) / len(tokens) >= 0.3


    def print_summary(self, baseline: List[Dict], ca: List[Dict]):
        n = len(baseline)
        sep = "═" * 68

        print(f"\n{sep}")
        print("  EVALUATION SUMMARY  —  RAGTruth Confidence-Aware RAG")
        print(sep)
        print(f"  Total questions : {n}")
        print(f"  Conf threshold  : {self.cfg.confidence_threshold}  "
              f"(high={self.cfg.high_conf_threshold})")
        print(f"  Conf method     : {self.cfg.confidence_method}")
        print()

        def _stats(results, label):
            total   = len(results)
            abstain = sum(1 for r in results if r.get("abstained"))
            hedged  = sum(1 for r in results if r.get("hedged"))
            ans     = total - abstain
            rouge   = (sum(r["rouge1"] for r in results if not r.get("abstained"))
                       / max(ans, 1))
            sup_lst = [r for r in results if r.get("supported") is not None]
            sup     = sum(1 for r in sup_lst if r["supported"]) / max(len(sup_lst), 1)
            conf    = sum(r["confidence"] for r in results) / total
            ood     = [r for r in results if r.get("domain") == "ood"]
            ood_abs = sum(1 for r in ood if r.get("abstained"))

            print(f"  {label}:")
            print(f"    Avg ROUGE-1 F1      : {rouge:.4f}  (answered qs only)")
            print(f"    Retrieval support   : {sup:.2%}  (answered qs only)")
            print(f"    Avg confidence      : {conf:.4f}")
            print(f"    Abstention rate     : {abstain/total:.2%}  ({abstain}/{total})")
            print(f"    Hedged rate         : {hedged/total:.2%}  ({hedged}/{total})")
            print(f"    Answer coverage     : {ans/total:.2%}  ({ans}/{total})")
            print(f"    OOD abstention      : {ood_abs}/{len(ood)}  "
                  f"({ood_abs/max(len(ood),1):.0%})")
            print()
            return rouge, sup, conf

        b_r, b_s, b_c = _stats(baseline, "BASELINE RAG")
        c_r, c_s, c_c = _stats(ca,       "CONFIDENCE-AWARE RAG")

        print("  IMPROVEMENT (answered questions):")
        print(f"    ROUGE-1 delta       : {c_r - b_r:+.4f}")
        print(f"    Support rate Δ      : {c_s - b_s:+.2%}")
        print(sep)

        print("\n  SAMPLE RESULTS (first 10 questions):\n")
        print(f"  {'ID':<5} {'Conf':>6} {'Dec':<9} {'B-R1':>6} {'CA-R1':>6}  "
              f"{'Dom':<4}  Question")
        print("  " + "─" * 68)
        for b, c in list(zip(baseline, ca))[:10]:
            dec_short = {"ABSTAIN": "ABS", "HEDGE": "HED", "GENERATE": "GEN"}.get(
                c.get("decision", ""), c.get("decision", "")[:3])
            print(
                f"  {b['id']:<5} {b['confidence']:>6.3f} {dec_short:<9} "
                f"{b['rouge1']:>6.4f} {c['rouge1']:>6.4f}  "
                f"{b.get('domain','?'):<4}  {b['question'][:40]}"
            )
        print()


    def save(self, baseline: List[Dict], ca: List[Dict]):
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        with open(f"{self.cfg.output_dir}/baseline_results.json", "w") as f:
            json.dump(baseline, f, indent=2)
        with open(f"{self.cfg.output_dir}/confidence_aware_results.json", "w") as f:
            json.dump(ca, f, indent=2)

        # Comparison CSV
        csv_path = f"{self.cfg.output_dir}/comparison.csv"
        fields = ["id", "domain", "question", "confidence",
                  "b_rouge1", "ca_rouge1", "ca_decision",
                  "b_supported", "ca_supported",
                  "b_answer", "ca_answer", "reference"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for b, c in zip(baseline, ca):
                w.writerow({
                    "id":          b["id"],
                    "domain":      b.get("domain", ""),
                    "question":    b["question"],
                    "confidence":  b["confidence"],
                    "b_rouge1":    b["rouge1"],
                    "ca_rouge1":   c["rouge1"],
                    "ca_decision": c.get("decision", ""),
                    "b_supported": b.get("supported", ""),
                    "ca_supported":c.get("supported", ""),
                    "b_answer":    b["answer"][:120],
                    "ca_answer":   c["answer"][:120],
                    "reference":   b.get("reference", ""),
                })
        logger.info(f"Results saved to {self.cfg.output_dir}/  (CSV + JSON)")
