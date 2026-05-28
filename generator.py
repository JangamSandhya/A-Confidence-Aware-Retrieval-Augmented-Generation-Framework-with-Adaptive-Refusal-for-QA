

import logging
import re
from typing import List, Tuple

from config import RAGConfig
from ragtruth_corpus import Document

logger = logging.getLogger(__name__)


class RAGGenerator:

    STANDARD_PROMPT = (
        "Answer the question using ONLY the information in the provided passages. "
        "If the passages do not contain enough information, respond with "
        "'The provided context does not contain sufficient information to answer this question.'\n\n"
        "Passages:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    HEDGE_PROMPT = (
        "The retrieved passages may not fully address the question. "
        "Based on the partial evidence below, provide a cautious and hedged answer. "
        "Clearly indicate any uncertainty.\n\n"
        "Passages:\n{context}\n\n"
        "Question: {question}\n\n"
        "Cautious answer:"
    )

    SUMMARY_PROMPT = (
        "Summarise the following passage in 2-3 sentences, "
        "including only information that is explicitly stated.\n\n"
        "Passage:\n{context}\n\n"
        "Summary:"
    )

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._model    = None
        self._tokenizer = None
        self._backend  = "extractive"
        self._try_load_hf()


    def generate(
        self,
        question: str,
        retrieved: List[Tuple[Document, float]],
        hedged: bool = False,
        task: str = "qa",
    ) -> str:
        
        context = self._build_context(retrieved)

        if task == "summary":
            prompt = self.SUMMARY_PROMPT.format(context=context, question=question)
        elif hedged:
            prompt = self.HEDGE_PROMPT.format(context=context, question=question)
        else:
            prompt = self.STANDARD_PROMPT.format(context=context, question=question)

        if self._backend == "hf":
            return self._hf_generate(prompt)
        return self._extractive(question, retrieved)


    @staticmethod
    def _build_context(
        retrieved: List[Tuple[Document, float]],
        max_chars: int = 2000,
    ) -> str:
        parts, total = [], 0
        for doc, score in retrieved:
            snippet = f"[Passage – {doc.task.upper()} | score={score:.3f}]\n{doc.source_info}"
            if total + len(snippet) > max_chars:
                snippet = snippet[: max_chars - total]
                parts.append(snippet)
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n\n".join(parts)


    def _try_load_hf(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info(f"Loading generator: {self.cfg.generator_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.generator_model)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.generator_model)
            if self.cfg.generator_device != "cpu":
                self._model = self._model.to(self.cfg.generator_device)
            self._model.eval()
            self._backend = "hf"
            logger.info("HuggingFace generator ready.")
        except Exception as e:
            logger.warning(f"HuggingFace generator unavailable ({e}). "
                           "Using extractive fallback.")

    def _hf_generate(self, prompt: str) -> str:
        import torch
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.cfg.max_input_length,
            truncation=True,
        )
        if self.cfg.generator_device != "cpu":
            inputs = {k: v.to(self.cfg.generator_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_output_length,
                num_beams=self.cfg.num_beams,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature if self.cfg.do_sample else 1.0,
                early_stopping=True,
            )
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()


    @staticmethod
    def _extractive(
        question: str,
        retrieved: List[Tuple[Document, float]],
    ) -> str:
        
        if not retrieved:
            return "No relevant documents found in the knowledge base."

        q_toks = set(re.findall(r"[a-z]{3,}", question.lower()))
        best_sent, best_overlap, best_title = None, -1, ""

        for doc, _ in retrieved[:3]:
            sents = re.split(r"(?<=[.!?])\s+", doc.source_info)
            for sent in sents:
                s_toks  = set(re.findall(r"[a-z]{3,}", sent.lower()))
                overlap = len(q_toks & s_toks)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sent  = sent.strip()
                    best_title = doc.title[:60]

        if best_sent:
            return f"{best_sent}  (Source: {best_title})"
        top = retrieved[0][0]
        return f"{top.source_info[:280]}…  (Source: {top.title[:60]})"
