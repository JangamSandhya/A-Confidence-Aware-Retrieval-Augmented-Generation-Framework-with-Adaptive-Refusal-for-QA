

import json
import logging
import os
import re
from collections import defaultdict
from typing import List, Dict, Optional

from config import RAGConfig
from pipeline import ConfidenceAwareRAGPipeline, RAGResult

logger = logging.getLogger(__name__)



DEFAULT_EVAL_QUESTIONS = [
    # Diabetes
    {
        "id": "q01",
        "question": "What is the first-line treatment for type 2 diabetes?",
        "reference": "Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus.",
    },
    {
        "id": "q02",
        "question": "What are the side effects of metformin?",
        "reference": "Common adverse effects include gastrointestinal discomfort, nausea, and diarrhoea. Lactic acidosis is a rare but serious complication.",
    },
    {
        "id": "q03",
        "question": "At what eGFR is metformin contraindicated?",
        "reference": "Metformin is contraindicated in patients with an eGFR below 30 mL/min/1.73 m².",
    },
    # COVID-19 
    {
        "id": "q04",
        "question": "How does SARS-CoV-2 enter human cells?",
        "reference": "SARS-CoV-2 infects human cells through binding of its spike protein to the ACE2 receptor.",
    },
    {
        "id": "q05",
        "question": "Which SARS-CoV-2 mutations enhance ACE2 binding?",
        "reference": "Mutations such as N501Y and E484K enhance ACE2 binding affinity.",
    },
    # Hypertension 
    {
        "id": "q06",
        "question": "What is the recommended blood pressure target according to ESC/ESH guidelines?",
        "reference": "The target blood pressure is below 130/80 mmHg for most adults.",
    },
    {
        "id": "q07",
        "question": "What are first-line antihypertensive agents?",
        "reference": "ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics are first-line agents.",
    },
    # Vaccines 
    {
        "id": "q08",
        "question": "How do mRNA vaccines work?",
        "reference": "mRNA vaccines encode the antigen; ribosomes translate the mRNA into the target protein, triggering humoral and cellular immune responses.",
    },
    {
        "id": "q09",
        "question": "What efficacy did BNT162b2 demonstrate?",
        "reference": "BNT162b2 demonstrated over 90% efficacy against the ancestral SARS-CoV-2 strain.",
    },
    # Alzheimer's 
    {
        "id": "q10",
        "question": "What is the amyloid cascade hypothesis?",
        "reference": "Abnormal accumulation of amyloid-beta peptides initiates a pathological cascade leading to Alzheimer's disease.",
    },
    {
        "id": "q11",
        "question": "What anti-amyloid drug received FDA approval in 2023?",
        "reference": "Lecanemab, an anti-amyloid monoclonal antibody, received FDA accelerated approval in 2023.",
    },
    #  CRISPR 
    {
        "id": "q12",
        "question": "What is CRISPR-Cas9 and how does it work?",
        "reference": "A guide RNA directs Cas9 to a specific genomic locus where it introduces a double-strand break, repaired by NHEJ or HDR.",
    },
    {
        "id": "q13",
        "question": "What are base editing and prime editing?",
        "reference": "Base editing and prime editing are refined CRISPR variants that minimise double-strand breaks.",
    },
    #  Atrial Fibrillation 
    {
        "id": "q14",
        "question": "What is the stroke risk associated with atrial fibrillation?",
        "reference": "AF significantly increases stroke risk due to stasis-mediated thrombus formation in the left atrial appendage.",
    },
    {
        "id": "q15",
        "question": "Which anticoagulants are recommended for non-valvular AF?",
        "reference": "DOACs such as apixaban or rivaroxaban are recommended for patients with non-valvular AF and CHA₂DS₂-VASc score ≥2.",
    },
    #  Oncology 
    {
        "id": "q16",
        "question": "What are PD-1 and CTLA-4 checkpoint inhibitors?",
        "reference": "Immune checkpoint inhibitors restore anti-tumour T-cell activity by targeting inhibitory receptors such as PD-1, PD-L1, and CTLA-4.",
    },
    {
        "id": "q17",
        "question": "What biomarkers predict ICI response in lung cancer?",
        "reference": "Tumour mutational burden and PD-L1 expression are predictive biomarkers for checkpoint inhibitor response.",
    },
    #  Sepsis 
    {
        "id": "q18",
        "question": "How is sepsis defined by Sepsis-3 criteria?",
        "reference": "Sepsis is life-threatening organ dysfunction caused by a dysregulated host response to infection, identified by SOFA score increase of ≥2.",
    },
    {
        "id": "q19",
        "question": "What is the first-choice vasopressor in septic shock?",
        "reference": "Norepinephrine is the first-choice vasopressor in septic shock management.",
    },
    #  Type 1 Diabetes 
    {
        "id": "q20",
        "question": "What causes type 1 diabetes mellitus?",
        "reference": "Type 1 diabetes results from autoimmune destruction of pancreatic beta cells, leading to absolute insulin deficiency.",
    },
    #  STEMI 
    {
        "id": "q21",
        "question": "What is the preferred reperfusion strategy for STEMI?",
        "reference": "Primary percutaneous coronary intervention within 90 minutes of first medical contact is the preferred reperfusion strategy.",
    },
    # BRCA 
    {
        "id": "q22",
        "question": "Why are PARP inhibitors effective in BRCA-mutated cancers?",
        "reference": "Loss of BRCA function renders tumours sensitive to PARP inhibitors due to synthetic lethality.",
    },
    #  COPD 
    {
        "id": "q23",
        "question": "What is the most effective intervention to slow COPD progression?",
        "reference": "Smoking cessation is the most effective intervention to slow COPD disease progression.",
    },
    #  CAR-T 
    {
        "id": "q24",
        "question": "What adverse effects are associated with CAR-T therapy?",
        "reference": "Cytokine release syndrome and immune effector cell-associated neurotoxicity syndrome are major adverse effects of CAR-T therapy.",
    },
    #  Stroke 
    {
        "id": "q25",
        "question": "Within what timeframe should tPA be administered for ischaemic stroke?",
        "reference": "Intravenous alteplase should be administered within 4.5 hours of symptom onset.",
    },
    # Gut microbiome 
    {
        "id": "q26",
        "question": "What are short-chain fatty acids and why are they beneficial?",
        "reference": "SCFAs produced by bacterial fermentation of dietary fibre improve insulin sensitivity and promote gut barrier integrity.",
    },
    #  Parkinson's 
    {
        "id": "q27",
        "question": "What is the gold standard pharmacological treatment for Parkinson's disease?",
        "reference": "Levodopa combined with a dopa-decarboxylase inhibitor (carbidopa) remains the gold standard treatment for Parkinson's disease.",
    },
    #  MS 
    {
        "id": "q28",
        "question": "What is the risk of PML with natalizumab?",
        "reference": "Progressive multifocal leukoencephalopathy caused by JC virus reactivation is a serious risk with natalizumab, especially in JC antibody positive patients with high index values.",
    },
    #  Heart Failure 
    {
        "id": "q29",
        "question": "What are the four pillars of guideline-directed therapy for HFrEF?",
        "reference": "ACE inhibitors/ARBs/ARNIs, beta-blockers, mineralocorticoid receptor antagonists, and SGLT2 inhibitors form the four pillars for HFrEF.",
    },
    #  Antibiotic resistance 
    {
        "id": "q30",
        "question": "What mechanisms confer antibiotic resistance?",
        "reference": "Resistance arises through enzymatic inactivation, target site modification, reduced outer membrane permeability, and efflux pump overexpression.",
    },
    #  Out-of-domain  
    {
        "id": "q31",
        "question": "What is the capital of France?",
        "reference": "N/A",
    },
    {
        "id": "q32",
        "question": "Explain quantum entanglement in simple terms.",
        "reference": "N/A",
    },
    {
        "id": "q33",
        "question": "Who wrote the novel War and Peace?",
        "reference": "N/A",
    },
    {
        "id": "q34",
        "question": "What are the best practices for software version control?",
        "reference": "N/A",
    },
]


class RAGEvaluator:

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg


    def load_questions(self) -> List[Dict]:
        path = self.cfg.eval_questions_path
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        logger.info("Using built-in evaluation question set (34 questions).")
        return DEFAULT_EVAL_QUESTIONS


    def evaluate_baseline(
        self,
        pipeline: ConfidenceAwareRAGPipeline,
        questions: List[Dict],
    ) -> List[Dict]:
        results = []
        for i, q_data in enumerate(questions, 1):
            q = q_data["question"]
            ref = q_data.get("reference", "")
            logger.info(f"  Baseline Q{i:02d}: {q[:60]}…")
            result = pipeline.baseline_answer(q)
            entry = result.to_dict()
            entry["id"] = q_data.get("id", str(i))
            entry["reference"] = ref
            entry["rouge1"] = self._rouge1(result.answer, ref)
            entry["has_retrieval_support"] = self._has_retrieval_support(
                result.answer, result.retrieved_docs
            )
            results.append(entry)
        return results

    def evaluate_confidence_aware(
        self,
        pipeline: ConfidenceAwareRAGPipeline,
        questions: List[Dict],
    ) -> List[Dict]:
        results = []
        for i, q_data in enumerate(questions, 1):
            q = q_data["question"]
            ref = q_data.get("reference", "")
            logger.info(f"  Conf-Aware Q{i:02d}: {q[:60]}…")
            result = pipeline.confidence_aware_answer(q)
            entry = result.to_dict()
            entry["id"] = q_data.get("id", str(i))
            entry["reference"] = ref
            entry["rouge1"] = self._rouge1(result.answer, ref) if not result.abstained else 0.0
            entry["has_retrieval_support"] = self._has_retrieval_support(
                result.answer, result.retrieved_docs
            ) if not result.abstained else None
            results.append(entry)
        return results


    def _rouge1(self, hypothesis: str, reference: str) -> float:
        """Unigram recall-precision F1 (ROUGE-1)."""
        if not reference or reference == "N/A":
            return 0.0
        h_tokens = set(re.findall(r"[a-z]+", hypothesis.lower()))
        r_tokens = set(re.findall(r"[a-z]+", reference.lower()))
        overlap = h_tokens & r_tokens
        if not overlap:
            return 0.0
        p = len(overlap) / len(h_tokens) if h_tokens else 0.0
        r = len(overlap) / len(r_tokens) if r_tokens else 0.0
        if p + r == 0:
            return 0.0
        return round(2 * p * r / (p + r), 4)

    def _has_retrieval_support(self, answer: str, retrieved_docs) -> bool:
       
        if not retrieved_docs:
            return False
        doc_text = " ".join(d.text for d, _ in retrieved_docs).lower()
        a_tokens = set(re.findall(r"[a-z]{4,}", answer.lower()))
        stop = {"that", "this", "with", "from", "have", "been", "they",
                "their", "when", "what", "which", "will", "more", "also",
                "note", "based", "answer", "only", "does", "information"}
        a_tokens -= stop
        if not a_tokens:
            return False
        matches = sum(1 for tok in a_tokens if tok in doc_text)
        return matches / len(a_tokens) >= 0.3


    def print_summary(self, baseline: List[Dict], ca: List[Dict]):
        n = len(baseline)
        print("\n" + "=" * 70)
        print("  EVALUATION SUMMARY")
        print("=" * 70)
        print(f"  Total questions: {n}")
        print()

        b_rouge = sum(r["rouge1"] for r in baseline) / n
        b_support = sum(1 for r in baseline if r.get("has_retrieval_support")) / n
        b_conf = sum(r["confidence"] for r in baseline) / n
        print("  BASELINE RAG:")
        print(f"    Avg ROUGE-1 F1      : {b_rouge:.4f}")
        print(f"    Retrieval support   : {b_support:.2%}")
        print(f"    Avg confidence      : {b_conf:.4f}")
        print(f"    Abstention rate     : 0.00% (never abstains)")
        print()

        abstained = sum(1 for r in ca if r["abstained"])
        hedged = sum(1 for r in ca if r.get("hedged"))
        generated = n - abstained
        ca_rouge = sum(r["rouge1"] for r in ca if not r["abstained"])
        ca_rouge /= max(generated, 1)
        ca_support_list = [r for r in ca if r.get("has_retrieval_support") is not None]
        ca_support = sum(1 for r in ca_support_list if r["has_retrieval_support"])
        ca_support /= max(len(ca_support_list), 1)
        ca_conf = sum(r["confidence"] for r in ca) / n

        print("  CONFIDENCE-AWARE RAG:")
        print(f"    Avg ROUGE-1 F1      : {ca_rouge:.4f}  (answered qs only)")
        print(f"    Retrieval support   : {ca_support:.2%}  (answered qs only)")
        print(f"    Avg confidence      : {ca_conf:.4f}")
        print(f"    Abstention rate     : {abstained/n:.2%}  ({abstained}/{n})")
        print(f"    Hedged rate         : {hedged/n:.2%}  ({hedged}/{n})")
        print(f"    Answer coverage     : {generated/n:.2%}  ({generated}/{n})")
        print()
        print("  IMPROVEMENT:")
        print(f"    ROUGE-1 delta       : {ca_rouge - b_rouge:+.4f}")
        print(f"    Support rate delta  : {ca_support - b_support:+.2%}")
        print("=" * 70)

    def save_results(self, baseline: List[Dict], ca: List[Dict]):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        b_path = os.path.join(self.cfg.output_dir, "baseline_results.json")
        ca_path = os.path.join(self.cfg.output_dir, "confidence_aware_results.json")
        with open(b_path, "w") as f:
            json.dump(baseline, f, indent=2)
        with open(ca_path, "w") as f:
            json.dump(ca, f, indent=2)
        # Combined comparison CSV
        self._write_comparison_csv(baseline, ca)
        logger.info(f"Results saved to {self.cfg.output_dir}/")

    def _write_comparison_csv(self, baseline: List[Dict], ca: List[Dict]):
        import csv
        path = os.path.join(self.cfg.output_dir, "comparison.csv")
        fields = [
            "id", "question",
            "baseline_answer", "baseline_confidence", "baseline_rouge1", "baseline_support",
            "ca_answer", "ca_confidence", "ca_rouge1", "ca_abstained", "ca_hedged",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for b, c in zip(baseline, ca):
                writer.writerow({
                    "id": b.get("id", ""),
                    "question": b.get("question", ""),
                    "baseline_answer": b.get("answer", "")[:120],
                    "baseline_confidence": b.get("confidence", ""),
                    "baseline_rouge1": b.get("rouge1", ""),
                    "baseline_support": b.get("has_retrieval_support", ""),
                    "ca_answer": c.get("answer", "")[:120],
                    "ca_confidence": c.get("confidence", ""),
                    "ca_rouge1": c.get("rouge1", ""),
                    "ca_abstained": c.get("abstained", ""),
                    "ca_hedged": c.get("hedged", ""),
                })
        logger.info(f"Comparison CSV → {path}")
