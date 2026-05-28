

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class RAGConfig:
    corpus_source: str = "pubmed_sample"   
    corpus_path: Optional[str] = None      
    max_documents: int = 5000              

    dpr_ctx_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base"
    dpr_question_encoder: str = "facebook/dpr-question_encoder-single-nq-base"
    generator_model: str = "google/flan-t5-base"   
    max_input_length: int = 512
    max_output_length: int = 256
    generator_device: str = "cpu"          

    top_k: int = 5                         
    embedding_batch_size: int = 32
    embedding_dim: int = 768               

    faiss_index_type: str = "Flat"         
    faiss_nlist: int = 100                 
    index_path: str = "data/faiss.index"
    docs_path: str = "data/documents.json"

    confidence_method: str = "gap"        
    confidence_threshold: float = 0.15    
    abstention_message: str = (
        "I could not find sufficiently reliable information in the knowledge base "
        "to answer this question confidently. Please consult a domain expert."
    )

    synth_qa_per_doc: int = 2
    synth_max_docs: int = 500

    eval_questions_path: str = "data/eval_questions.json"
    output_dir: str = "results"

    num_beams: int = 4
    temperature: float = 1.0
    do_sample: bool = False

    def load_json(self, path: str):
        with open(path) as f:
            d = json.load(f)
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def __repr__(self):
        return f"RAGConfig({json.dumps(asdict(self), indent=2)})"
