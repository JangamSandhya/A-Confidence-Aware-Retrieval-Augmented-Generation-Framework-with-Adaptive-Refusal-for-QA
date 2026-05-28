

import json
import logging
import os
import re
from typing import Dict, List, Tuple

from config import RAGConfig
from ragtruth_corpus import Document

logger = logging.getLogger(__name__)


# Template: (regex, question_fn, answer_fn)
TEMPLATES: List[Tuple] = [
    # "X is Y"
    (r"^(.+?) is (?:a |an |the )?(.+?)\.",
     lambda m: f"What is {m.group(1).strip()}?",
     lambda m: m.group(0)),
    # "X are Y"
    (r"^(.+?) are (?:a |an |the )?(.+?)\.",
     lambda m: f"What are {m.group(1).strip()}?",
     lambda m: m.group(0)),
    # "X was introduced by Y"
    (r"^(.+?) was introduced by (.+?)[,.]",
     lambda m: f"Who introduced {m.group(1).strip()}?",
     lambda m: m.group(0)),
    # "X uses Y"
    (r"^(.+?) uses? (.+?)\.",
     lambda m: f"What does {m.group(1).strip()} use?",
     lambda m: m.group(0)),
    # "X enables Y"
    (r"^(.+?) enables? (.+?)\.",
     lambda m: f"What does {m.group(1).strip()} enable?",
     lambda m: m.group(0)),
    # "X improves Y"
    (r"^(.+?) (?:improves?|reduces?|increases?) (.+?)\.",
     lambda m: f"What does {m.group(1).strip()} affect?",
     lambda m: m.group(0)),
    # "X consists of Y"
    (r"^(.+?) consists? of (.+?)\.",
     lambda m: f"What does {m.group(1).strip()} consist of?",
     lambda m: m.group(0)),
    # "X supports Y"
    (r"^(.+?) supports? (.+?)\.",
     lambda m: f"What does {m.group(1).strip()} support?",
     lambda m: m.group(0)),
    # "X was trained on Y"
    (r"^(.+?) (?:is|was) trained on (.+?)\.",
     lambda m: f"What is {m.group(1).strip()} trained on?",
     lambda m: m.group(0)),
    # "X achieves Y"
    (r"^(.+?) achieves? (.+?)\.",
     lambda m: f"What does {m.group(1).strip()} achieve?",
     lambda m: m.group(0)),
]


class SyntheticQAGenerator:

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def generate(self, documents: List[Document]) -> List[Dict]:
        docs = documents[: self.cfg.synth_max_docs]
        pairs = []
        for doc in docs:
            doc_pairs = self._from_doc(doc)[: self.cfg.synth_qa_per_doc]
            pairs.extend(doc_pairs)
        logger.info(
            f"Synthetic QA: {len(pairs)} pairs from {len(docs)} documents."
        )
        return pairs

    def _from_doc(self, doc: Document) -> List[Dict]:
        sents  = self._split_sentences(doc.source_info)
        pairs  = []
        for sent in sents:
            q, a = self._apply_templates(sent.strip())
            if q and a:
                pairs.append({
                    "doc_id":          doc.id,
                    "doc_task":        doc.task,
                    "question":        q,
                    "answer":          a,
                    "source_sentence": sent.strip(),
                })
        if not pairs and sents:
            pairs.append({
                "doc_id":          doc.id,
                "doc_task":        doc.task,
                "question":        f"What does this {doc.task} passage describe?",
                "answer":          sents[0],
                "source_sentence": sents[0],
            })
        return pairs

    @staticmethod
    def _apply_templates(sentence: str) -> Tuple[str, str]:
        for pat, q_fn, a_fn in TEMPLATES:
            m = re.match(pat, sentence, re.IGNORECASE)
            if m:
                try:
                    q = q_fn(m).strip()
                    a = a_fn(m).strip()
                    if len(q) > 10 and len(a) > 15:
                        return q, a
                except Exception:
                    continue
        return "", ""

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]

    def save(self, pairs: List[Dict], path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(pairs, f, indent=2)
        logger.info(f"Synthetic QA saved → {path}  ({len(pairs)} pairs)")
