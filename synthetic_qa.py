

import logging
import re
import random
from typing import List, Dict, Tuple

from config import RAGConfig
from corpus import Document

logger = logging.getLogger(__name__)


class SyntheticQAGenerator:
   

    TEMPLATES = [
        # "X is Y" → "What is X?"
        (r"^(.+?) is (?:the |a |an )?(.+?)\.",
         lambda m: f"What is {m.group(1).strip()}?",
         lambda m: m.group(0)),
        # "X are Y" → "What are X?"
        (r"^(.+?) are (?:the |a |an )?(.+?)\.",
         lambda m: f"What are {m.group(1).strip()}?",
         lambda m: m.group(0)),
        # "X involves Y" → "What does X involve?"
        (r"^(.+?) involves? (.+?)\.",
         lambda m: f"What does {m.group(1).strip()} involve?",
         lambda m: m.group(0)),
        # "X requires Y" → "What does X require?"
        (r"^(.+?) requires? (.+?)\.",
         lambda m: f"What does {m.group(1).strip()} require?",
         lambda m: m.group(0)),
        # "X reduces Y" → "What does X reduce?"
        (r"^(.+?) reduces? (.+?)\.",
         lambda m: f"What does {m.group(1).strip()} reduce?",
         lambda m: m.group(0)),
        # "The treatment of X includes Y" style
        (r"^(?:The )?treatment (?:of|for) (.+?) (?:includes?|involves?) (.+?)\.",
         lambda m: f"What is the treatment for {m.group(1).strip()}?",
         lambda m: m.group(0)),
        # "X is approved for Y" → "What is X approved for?"
        (r"^(.+?) (?:is|are) approved for (.+?)\.",
         lambda m: f"What is {m.group(1).strip()} approved for?",
         lambda m: m.group(0)),
        # "X is caused by Y" → "What causes X?"
        (r"^(.+?) (?:is|are) caused by (.+?)\.",
         lambda m: f"What causes {m.group(1).strip()}?",
         lambda m: m.group(0)),
    ]

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def generate(self, documents: List[Document]) -> List[Dict]:
        random.seed(42)
        docs = documents[: self.cfg.synth_max_docs]
        all_pairs = []
        for doc in docs:
            pairs = self._extract_pairs(doc)
            # limit per doc
            pairs = pairs[: self.cfg.synth_qa_per_doc]
            all_pairs.extend(pairs)

        logger.info(f"Synthetic QA: generated {len(all_pairs)} pairs from {len(docs)} docs.")
        return all_pairs

    def _extract_pairs(self, doc: Document) -> List[Dict]:
        sentences = self._split_sentences(doc.text)
        pairs = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 30:
                continue
            q, a = self._apply_templates(sent)
            if q:
                pairs.append({
                    "doc_id": doc.id,
                    "doc_title": doc.title,
                    "question": q,
                    "answer": a,
                    "source_sentence": sent,
                })
        if not pairs and sentences:
            pairs.append({
                "doc_id": doc.id,
                "doc_title": doc.title,
                "question": f"What is described in the context of {doc.title}?",
                "answer": sentences[0],
                "source_sentence": sentences[0],
            })
        return pairs

    def _apply_templates(self, sentence: str) -> Tuple[str, str]:
        for pattern, q_fn, a_fn in self.TEMPLATES:
            m = re.match(pattern, sentence, re.IGNORECASE)
            if m:
                try:
                    q = q_fn(m)
                    a = a_fn(m)
                    if len(q) > 10 and len(a) > 10:
                        return q, a
                except Exception:
                    continue
        return "", ""

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]




class LLMSyntheticQAGenerator:
    
   

    PROMPT = (
        "Generate a factual question-answer pair from the following passage.\n\n"
        "Passage: {passage}\n\n"
        "Format:\nQuestion: <question>\nAnswer: <answer>"
    )

    def __init__(self, generator):
        self.generator = generator   # RAGGenerator instance

    def generate_from_doc(self, doc: Document, n: int = 2) -> List[Dict]:
        pairs = []
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.text) if len(s) > 50]
        random.shuffle(sents)
        for sent in sents[:n]:
            prompt = self.PROMPT.format(passage=sent)
            output = self.generator._hf_generate(prompt)
            q, a = self._parse_output(output)
            if q and a:
                pairs.append({
                    "doc_id": doc.id,
                    "doc_title": doc.title,
                    "question": q,
                    "answer": a,
                    "source_sentence": sent,
                })
        return pairs

    @staticmethod
    def _parse_output(text: str) -> Tuple[str, str]:
        q_match = re.search(r"Question:\s*(.+)", text)
        a_match = re.search(r"Answer:\s*(.+)", text)
        q = q_match.group(1).strip() if q_match else ""
        a = a_match.group(1).strip() if a_match else ""
        return q, a
