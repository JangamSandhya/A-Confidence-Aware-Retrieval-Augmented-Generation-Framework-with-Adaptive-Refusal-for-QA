

import logging
import re
from typing import List, Tuple, Optional

from config import RAGConfig
from corpus import Document

logger = logging.getLogger(__name__)


class RAGGenerator:

    PROMPT_TEMPLATE = (
        "Answer the following question based ONLY on the provided context. "
        "If the context does not contain enough information, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    HEDGE_TEMPLATE = (
        "The retrieved documents may not fully address this question. "
        "Based on partial evidence from the knowledge base:\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a cautious, hedged answer:"
    )

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._backend = None
        self._try_load_model()


    def generate(
        self,
        question: str,
        retrieved_docs: List[Tuple[Document, float]],
        hedged: bool = False,
    ) -> str:
        
        
        context = self._build_context(retrieved_docs)
        template = self.HEDGE_TEMPLATE if hedged else self.PROMPT_TEMPLATE
        prompt = template.format(context=context, question=question)

        if self._backend == "hf":
            return self._hf_generate(prompt)
        else:
            return self._extractive_fallback(question, retrieved_docs)


    def _build_context(
        self, retrieved: List[Tuple[Document, float]], max_chars: int = 1800
    ) -> str:
        parts = []
        total = 0
        for doc, score in retrieved:
            snippet = f"[{doc.title}]\n{doc.text}"
            if total + len(snippet) > max_chars:
                snippet = snippet[: max_chars - total]
                parts.append(snippet)
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n\n".join(parts)


    def _try_load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info(f"Loading generator: {self.cfg.generator_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.generator_model)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.generator_model)
            if self.cfg.generator_device != "cpu":
                import torch
                self._model = self._model.to(self.cfg.generator_device)
            self._model.eval()
            self._backend = "hf"
            logger.info("Generator ready (HuggingFace).")
        except Exception as e:
            logger.warning(f"HuggingFace generator unavailable ({e}). Using extractive fallback.")
            self._backend = "extractive"

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
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_output_length,
                num_beams=self.cfg.num_beams,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature if self.cfg.do_sample else 1.0,
                early_stopping=True,
            )
        answer = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer.strip()


    def _extractive_fallback(
        self,
        question: str,
        retrieved: List[Tuple[Document, float]],
    ) -> str:
        
        if not retrieved:
            return "No relevant documents were found."

        q_tokens = set(re.findall(r"[a-z]+", question.lower()))
        best_sent = None
        best_overlap = -1

        for doc, _ in retrieved[:2]:
            sentences = re.split(r"(?<=[.!?])\s+", doc.text)
            for sent in sentences:
                s_tokens = set(re.findall(r"[a-z]+", sent.lower()))
                overlap = len(q_tokens & s_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sent = sent

        if best_sent:
            return f"{best_sent} (Source: {retrieved[0][0].title})"
        return retrieved[0][0].text[:300] + "…"
