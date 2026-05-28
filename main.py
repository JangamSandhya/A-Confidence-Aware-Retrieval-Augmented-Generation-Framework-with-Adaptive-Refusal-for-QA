

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def check_torch_available():
    try:
        import torch
        from transformers import DPRQuestionEncoder
        return True
    except ImportError:
        return False


def run_lightweight_pipeline(mode: str, cfg_path: str = None):
    from config import RAGConfig
    from corpus import CorpusLoader
    from demo import (TFIDFEncoder, NumpyRetriever, extractive_generate,
                      compute_confidence, rouge1, has_retrieval_support,
                      print_summary, write_comparison_csv)
    from evaluator import DEFAULT_EVAL_QUESTIONS
    from synthetic_qa import SyntheticQAGenerator
    import numpy as np

    cfg = RAGConfig()
    if cfg_path and os.path.exists(cfg_path):
        cfg.load_json(cfg_path)

    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info("Loading corpus...")
    loader = CorpusLoader(cfg)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents.")

    if mode == "synth":
        logger.info("=== Generating Synthetic QA Pairs ===")
        generator = SyntheticQAGenerator(cfg)
        qa_pairs = generator.generate(docs)
        out_path = os.path.join(cfg.output_dir, "synthetic_qa.json")
        with open(out_path, "w") as f:
            json.dump(qa_pairs, f, indent=2)
        logger.info(f"Generated {len(qa_pairs)} QA pairs -> {out_path}")
        return

    logger.info("Building TF-IDF index...")
    encoder = TFIDFEncoder(dim=128)
    retriever = NumpyRetriever(encoder)
    retriever.index(docs)

    questions = DEFAULT_EVAL_QUESTIONS
    if cfg.eval_questions_path and os.path.exists(cfg.eval_questions_path):
        with open(cfg.eval_questions_path) as f:
            questions = json.load(f)
    logger.info(f"Evaluating on {len(questions)} questions.")

    THRESHOLD = cfg.confidence_threshold
    TOP_K = cfg.top_k
    METHOD = cfg.confidence_method
    ABSTENTION_MSG = cfg.abstention_message

    baseline_results = []
    ca_results = []

    for i, q_data in enumerate(questions, 1):
        q = q_data["question"]
        ref = q_data.get("reference", "N/A")
        logger.info(f"  Q{i:02d}: {q[:60]}...")

        retrieved, all_scores = retriever.search(q, TOP_K)
        top_scores = [s for _, s in retrieved]
        conf = compute_confidence(top_scores, all_scores, METHOD)

        b_answer = extractive_generate(q, retrieved)
        b_rouge = rouge1(b_answer, ref)
        b_support = has_retrieval_support(b_answer, retrieved)

        baseline_results.append({
            "id": q_data.get("id", str(i)),
            "question": q,
            "answer": b_answer,
            "confidence": round(conf, 4),
            "abstained": False,
            "hedged": False,
            "rouge1": b_rouge,
            "has_retrieval_support": b_support,
            "reference": ref,
            "top_doc": retrieved[0][0].title if retrieved else "N/A",
        })

        abstained = False
        hedged = False
        if conf < THRESHOLD:
            ca_answer = ABSTENTION_MSG
            abstained = True
        elif conf < 0.5:
            ca_answer = (f"[Note: Moderate retrieval confidence ({conf:.2f})] "
                         + extractive_generate(q, retrieved))
            hedged = True
        else:
            ca_answer = extractive_generate(q, retrieved)

        ca_rouge = rouge1(ca_answer, ref) if not abstained else 0.0
        ca_support = has_retrieval_support(ca_answer, retrieved) if not abstained else None

        ca_results.append({
            "id": q_data.get("id", str(i)),
            "question": q,
            "answer": ca_answer,
            "confidence": round(conf, 4),
            "abstained": abstained,
            "hedged": hedged,
            "rouge1": ca_rouge,
            "has_retrieval_support": ca_support,
            "reference": ref,
            "top_doc": retrieved[0][0].title if retrieved else "N/A",
        })

    with open(os.path.join(cfg.output_dir, "baseline_results.json"), "w") as f:
        json.dump(baseline_results, f, indent=2)
    with open(os.path.join(cfg.output_dir, "confidence_aware_results.json"), "w") as f:
        json.dump(ca_results, f, indent=2)

    write_comparison_csv(baseline_results, ca_results)
    print_summary(baseline_results, ca_results, THRESHOLD)
    logger.info(f"Results saved to {cfg.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Confidence-Aware RAG System")
    parser.add_argument(
        "--mode", choices=["build", "synth", "eval", "all"], default="all",
        help="build=index, synth=synthetic QA, eval=evaluate, all=full pipeline"
    )
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    args = parser.parse_args()

    logger.info("Using lightweight TF-IDF + numpy pipeline.")
    run_lightweight_pipeline(args.mode, args.config)


if __name__ == "__main__":
    main()
