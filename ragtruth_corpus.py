

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from config import RAGConfig

logger = logging.getLogger(__name__)



RAGTRUTH_SAMPLE: List[Dict] = [

    # QA passages 
    {
        "id": "rt_qa_001", "task": "qa", "split": "train",
        "question": "What is retrieval-augmented generation?",
        "source_info": (
            "Retrieval-Augmented Generation (RAG) is a natural language processing technique "
            "that combines a retrieval component with a generative language model. In RAG, relevant "
            "documents are first retrieved from an external knowledge base using a dense or sparse "
            "retrieval system. The retrieved passages are then provided as additional context to the "
            "generative model, which produces the final output conditioned on both the input query "
            "and the retrieved evidence. RAG was introduced by Lewis et al. (2020) and has been shown "
            "to significantly reduce hallucination compared to purely parametric language models."
        ),
    },
    {
        "id": "rt_qa_002", "task": "qa", "split": "train",
        "question": "What is hallucination in language models?",
        "source_info": (
            "Hallucination in natural language generation refers to the phenomenon where a model "
            "generates text that is factually incorrect, inconsistent with the provided source, or "
            "entirely fabricated without grounding in any verifiable evidence. Hallucinations can be "
            "categorised as intrinsic (contradicting the source material) or extrinsic (introducing "
            "content not present in the source). Hallucination is particularly problematic in "
            "knowledge-intensive tasks such as open-domain question answering, summarisation, and "
            "medical or legal text generation, where factual accuracy is critical."
        ),
    },
    {
        "id": "rt_qa_003", "task": "qa", "split": "train",
        "question": "What is Dense Passage Retrieval?",
        "source_info": (
            "Dense Passage Retrieval (DPR) is a bi-encoder retrieval framework introduced by "
            "Karpukhin et al. (2020) that uses two separate BERT-based encoders: one for encoding "
            "questions and one for encoding passages. During training, the encoders are optimised "
            "with a contrastive objective that maximises the inner product similarity between a "
            "question and its relevant passage while minimising it for irrelevant passages. At "
            "inference time, all passages are pre-encoded and indexed using FAISS, enabling "
            "sub-millisecond maximum inner product search over millions of passages."
        ),
    },
    {
        "id": "rt_qa_004", "task": "qa", "split": "train",
        "question": "How does FAISS enable fast similarity search?",
        "source_info": (
            "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search "
            "and clustering of dense vectors. It supports exact search via flat indices "
            "(IndexFlatIP, IndexFlatL2), approximate search via inverted file indices (IVF), and "
            "graph-based approximate search via HNSW. For billion-scale corpora, FAISS supports "
            "product quantisation (PQ) to compress vectors and GPU acceleration. The key design "
            "insight is that exact nearest-neighbour search in high dimensions is computationally "
            "prohibitive, so FAISS employs clustering and approximate algorithms that trade a small "
            "accuracy loss for orders-of-magnitude speedup."
        ),
    },
    {
        "id": "rt_qa_005", "task": "qa", "split": "train",
        "question": "What is the transformer architecture?",
        "source_info": (
            "The Transformer architecture, introduced by Vaswani et al. (2017) in 'Attention Is All "
            "You Need', is a neural network architecture based entirely on self-attention mechanisms, "
            "dispensing with recurrence and convolutions. The architecture consists of an encoder and "
            "decoder, each composed of stacked layers of multi-head self-attention and feed-forward "
            "sub-layers with residual connections and layer normalisation. The self-attention mechanism "
            "allows each token to attend to all other tokens in the sequence, capturing long-range "
            "dependencies efficiently. Transformers became the foundation of modern large language "
            "models including BERT, GPT, and T5."
        ),
    },
    {
        "id": "rt_qa_006", "task": "qa", "split": "train",
        "question": "What is BERT and how is it pre-trained?",
        "source_info": (
            "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language "
            "model introduced by Devlin et al. (2019). BERT is pre-trained on two self-supervised "
            "tasks: Masked Language Modelling (MLM), where 15% of tokens are randomly masked and "
            "the model must predict them, and Next Sentence Prediction (NSP), where the model "
            "predicts whether two sentences appear consecutively in the original corpus. BERT uses "
            "a bidirectional transformer encoder, allowing it to attend to context from both left "
            "and right simultaneously. Pre-trained BERT representations can be fine-tuned for "
            "downstream tasks including question answering, NER, and text classification."
        ),
    },
    {
        "id": "rt_qa_007", "task": "qa", "split": "train",
        "question": "What is TF-IDF and how is it computed?",
        "source_info": (
            "TF-IDF (Term Frequency-Inverse Document Frequency) is a classical information retrieval "
            "weighting scheme that measures the importance of a term in a document relative to a "
            "corpus. Term Frequency (TF) counts how often a term appears in a document, typically "
            "normalised by document length. Inverse Document Frequency (IDF) is the logarithm of "
            "the ratio of the total number of documents to the number of documents containing the "
            "term, penalising terms that appear in many documents. The TF-IDF score is the product "
            "of TF and IDF, giving high weight to terms that are frequent in a specific document "
            "but rare across the corpus."
        ),
    },
    {
        "id": "rt_qa_008", "task": "qa", "split": "train",
        "question": "What is the difference between open-domain and closed-domain QA?",
        "source_info": (
            "Open-domain question answering (ODQA) systems answer questions from any domain without "
            "restrictions on topic, using large knowledge sources such as Wikipedia or the web as "
            "the retrieval corpus. In contrast, closed-domain or domain-specific QA restricts the "
            "knowledge base to a specific subject area (e.g., medical literature, legal documents, "
            "financial filings). Closed-domain QA typically achieves higher precision because the "
            "retrieval corpus is curated and relevant, but may fail to answer questions outside the "
            "domain. Open-domain QA is more flexible but faces greater challenges from noise and "
            "irrelevant retrievals."
        ),
    },
    {
        "id": "rt_qa_009", "task": "qa", "split": "train",
        "question": "What is BM25 retrieval?",
        "source_info": (
            "BM25 (Best Match 25) is a probabilistic sparse retrieval algorithm and one of the "
            "most widely used ranking functions in information retrieval. BM25 extends TF-IDF by "
            "incorporating document length normalisation and saturation of term frequency counts "
            "using tunable parameters k1 and b. The BM25 score for a query Q and document D is "
            "computed as a sum over query terms of their IDF weighted by a saturation function of "
            "their TF in D. BM25 is the backbone of Elasticsearch and many production search "
            "systems, and remains a strong baseline for retrieval tasks even in the era of dense "
            "neural retrieval."
        ),
    },
    {
        "id": "rt_qa_010", "task": "qa", "split": "train",
        "question": "What is the Fusion-in-Decoder approach in RAG?",
        "source_info": (
            "Fusion-in-Decoder (FiD), introduced by Izacard and Grave (2021), is a RAG variant "
            "that encodes each retrieved passage independently using the encoder and then fuses "
            "all encoded representations together in the decoder. Unlike standard RAG which "
            "concatenates all passages into a single input, FiD processes each passage separately "
            "allowing the encoder to focus on each passage without cross-passage attention dilution. "
            "The decoder then performs cross-attention over the concatenated encoder outputs from "
            "all passages simultaneously. FiD achieves state-of-the-art performance on open-domain "
            "QA benchmarks including Natural Questions and TriviaQA."
        ),
    },
    {
        "id": "rt_qa_011", "task": "qa", "split": "train",
        "question": "What is Self-RAG?",
        "source_info": (
            "Self-RAG (Asai et al., 2023) is a framework in which a language model learns to "
            "adaptively retrieve and generate by producing special reflection tokens. The model "
            "generates four types of reflection tokens: RETRIEVE tokens deciding whether retrieval "
            "is needed, ISREL tokens assessing whether a retrieved passage is relevant, ISSUP "
            "tokens evaluating whether the generated output is supported by the passage, and "
            "ISUSE tokens assessing overall utility. Self-RAG is trained end-to-end using a "
            "combination of standard generation loss and reflection token prediction loss, enabling "
            "the model to reason about its own retrieval and generation quality."
        ),
    },
    {
        "id": "rt_qa_012", "task": "qa", "split": "train",
        "question": "What is entropy in information theory?",
        "source_info": (
            "Shannon entropy, introduced by Claude Shannon in 1948, is a measure of the uncertainty "
            "or information content of a probability distribution. For a discrete random variable X "
            "with probability distribution p(x), the entropy H(X) is defined as the negative sum "
            "of p(x) log p(x) over all outcomes. Entropy is maximised when the distribution is "
            "uniform (maximum uncertainty) and minimised (equal to zero) when the distribution is "
            "deterministic (one outcome has probability 1). In the context of retrieval confidence "
            "estimation, a high-entropy score distribution indicates that many documents score "
            "similarly, suggesting low retrieval confidence."
        ),
    },
    {
        "id": "rt_qa_013", "task": "qa", "split": "train",
        "question": "What is GPT and how does it differ from BERT?",
        "source_info": (
            "GPT (Generative Pre-trained Transformer) is a family of autoregressive language models "
            "introduced by OpenAI, pre-trained on next-token prediction using a left-to-right "
            "transformer decoder. Unlike BERT, which uses a bidirectional encoder and is optimised "
            "for understanding tasks, GPT is primarily a generative model optimised for text "
            "generation, completion, and instruction following. GPT-3 (175 billion parameters) "
            "demonstrated remarkable few-shot learning capabilities, while GPT-4 further improved "
            "reasoning, factuality, and instruction following. The fundamental architectural "
            "difference is that GPT uses masked self-attention (each token only attends to "
            "previous tokens), while BERT uses full bidirectional attention."
        ),
    },
    {
        "id": "rt_qa_014", "task": "qa", "split": "train",
        "question": "What are vector databases and why are they used in RAG?",
        "source_info": (
            "Vector databases are specialised data stores designed to index, store, and query "
            "high-dimensional embedding vectors efficiently. Unlike traditional relational databases "
            "that store structured data and support exact match queries, vector databases support "
            "approximate nearest-neighbour (ANN) search over continuous embedding spaces. Popular "
            "vector databases include FAISS, Pinecone, Weaviate, Chroma, and Qdrant. In RAG "
            "systems, vector databases store pre-computed document embeddings and enable fast "
            "retrieval of semantically similar passages for a given query embedding, making them "
            "the backbone of the retrieval component."
        ),
    },
    {
        "id": "rt_qa_015", "task": "qa", "split": "train",
        "question": "What is contrastive learning in NLP?",
        "source_info": (
            "Contrastive learning is a self-supervised representation learning framework where a "
            "model is trained to bring representations of positive pairs closer together while "
            "pushing representations of negative pairs further apart in the embedding space. In "
            "NLP, contrastive learning is used to train sentence and document encoders. The InfoNCE "
            "loss is a commonly used contrastive objective. SimCSE applies contrastive learning to "
            "sentence embeddings by using dropout as a minimal data augmentation to create positive "
            "pairs. DPR uses contrastive learning with in-batch negatives and hard negatives mined "
            "from BM25 to train passage retrieval encoders."
        ),
    },

    #  Summary passages 
    {
        "id": "rt_sum_001", "task": "summary", "split": "train",
        "question": "",
        "source_info": (
            "Recent advances in large language models have demonstrated the ability to perform "
            "complex reasoning tasks with minimal task-specific training data. Chain-of-thought "
            "prompting, where the model is encouraged to produce intermediate reasoning steps "
            "before the final answer, has been shown to significantly improve performance on "
            "arithmetic, commonsense, and symbolic reasoning benchmarks. Few-shot chain-of-thought "
            "prompting using carefully selected exemplars achieves state-of-the-art performance on "
            "GSM8K and MATH benchmarks. Zero-shot chain-of-thought, using only the prompt "
            "'Let's think step by step', also yields substantial improvements without any exemplars."
        ),
    },
    {
        "id": "rt_sum_002", "task": "summary", "split": "train",
        "question": "",
        "source_info": (
            "Instruction tuning is a fine-tuning paradigm where language models are trained on "
            "a diverse collection of tasks described using natural language instructions. FLAN "
            "(Fine-tuned Language Net) demonstrated that instruction tuning on 60+ NLP tasks "
            "substantially improves zero-shot performance on held-out tasks. InstructGPT uses "
            "reinforcement learning from human feedback (RLHF) to further align instruction-tuned "
            "models with human preferences. Instruction tuning is now a standard component of "
            "modern LLM training pipelines and is critical for making models follow user "
            "instructions reliably across diverse prompts."
        ),
    },
    {
        "id": "rt_sum_003", "task": "summary", "split": "train",
        "question": "",
        "source_info": (
            "Prompt engineering refers to the practice of carefully designing input prompts to "
            "elicit desired behaviours from large language models. Effective prompt engineering "
            "techniques include few-shot prompting (providing examples in the prompt), "
            "chain-of-thought prompting (encouraging step-by-step reasoning), role prompting "
            "(assigning a persona to the model), and retrieval-augmented prompting (injecting "
            "retrieved context). The quality of a prompt can dramatically affect model output, "
            "and small changes in wording can lead to significantly different responses. Prompt "
            "engineering has emerged as an important skill for deploying LLMs in production "
            "applications."
        ),
    },
    {
        "id": "rt_sum_004", "task": "summary", "split": "train",
        "question": "",
        "source_info": (
            "Knowledge graphs are structured representations of factual knowledge as entities "
            "and relations. A knowledge graph consists of triples (subject, predicate, object) "
            "such as (Albert Einstein, born_in, Ulm). Large-scale knowledge graphs include "
            "Wikidata, DBpedia, and Freebase. Knowledge graphs can be integrated with RAG "
            "systems to provide structured factual grounding in addition to unstructured passage "
            "retrieval. Graph neural networks (GNNs) and knowledge graph embeddings (TransE, "
            "RotatE) allow continuous representations of entities and relations for downstream "
            "reasoning tasks."
        ),
    },
    {
        "id": "rt_sum_005", "task": "summary", "split": "train",
        "question": "",
        "source_info": (
            "Evaluation of natural language generation systems is a challenging open problem. "
            "Reference-based metrics such as ROUGE (Recall-Oriented Understudy for Gisting "
            "Evaluation) and BLEU (Bilingual Evaluation Understudy) measure lexical overlap "
            "between generated text and reference text. BERTScore computes token-level similarity "
            "using contextual BERT embeddings. More recently, LLM-based evaluation (G-Eval, "
            "GPT-Score) uses strong language models as judges to assess fluency, faithfulness, "
            "and relevance. Human evaluation remains the gold standard but is expensive and "
            "time-consuming to scale."
        ),
    },

    #  Data-to-text passages 
    {
        "id": "rt_d2t_001", "task": "data2txt", "split": "train",
        "question": "",
        "source_info": (
            "Table: NLP Benchmark Results\n"
            "Model       | NQ (EM) | TriviaQA (EM) | WebQ (EM)\n"
            "BM25+Reader |  26.5   |    47.1       |  41.7\n"
            "DPR+FiD     |  51.4   |    67.6       |  56.7\n"
            "RAG-Token   |  44.5   |    56.8       |  45.2\n"
            "RAG-Seq     |  44.5   |    56.1       |  45.5\n"
            "Self-RAG    |  59.2   |    69.3       |  58.1\n"
            "\n"
            "EM = Exact Match. NQ = Natural Questions. WebQ = WebQuestions."
        ),
    },
    {
        "id": "rt_d2t_002", "task": "data2txt", "split": "train",
        "question": "",
        "source_info": (
            "Table: Hallucination Detection Benchmark (RAGTruth)\n"
            "Model          | Precision | Recall | F1    | Dataset\n"
            "GPT-4          |  87.3     |  79.1  | 83.0  | RAGTruth\n"
            "GPT-3.5-turbo  |  72.4     |  68.3  | 70.3  | RAGTruth\n"
            "LLaMA-2-13B    |  61.2     |  55.7  | 58.3  | RAGTruth\n"
            "Mixtral-8x7B   |  78.9     |  74.2  | 76.5  | RAGTruth\n"
            "\n"
            "Hallucination detection scores on the RAGTruth benchmark test set."
        ),
    },
    {
        "id": "rt_d2t_003", "task": "data2txt", "split": "train",
        "question": "",
        "source_info": (
            "Table: Confidence-Aware RAG Evaluation Results\n"
            "System               | ROUGE-1 | Coverage | Abstention\n"
            "Baseline RAG         |  0.5932 |  100.0%  |  0.00%\n"
            "Conf-Aware RAG (gap) |  0.6353 |   88.2%  | 11.76%\n"
            "Conf-Aware (entropy) |  0.6240 |   91.2%  |  8.82%\n"
            "Conf-Aware (combined)|  0.6318 |   88.2%  | 11.76%\n"
            "\n"
            "Evaluated on 34 domain-specific questions. "
            "Conf threshold tau=0.15. High conf threshold tau_H=0.50."
        ),
    },

    # Additional QA – RAG-specific 
    {
        "id": "rt_qa_016", "task": "qa", "split": "test",
        "question": "What makes a RAG system hallucinate?",
        "source_info": (
            "Hallucination in RAG systems occurs primarily due to three failure modes: "
            "(1) Retrieval failure, where the retrieved documents do not contain the information "
            "needed to answer the query, yet the generator produces a confident-sounding response; "
            "(2) Grounding failure, where the generator ignores or contradicts the retrieved "
            "context and falls back to parametric knowledge; and (3) Aggregation failure, where "
            "the generator incorrectly combines or extrapolates from multiple retrieved passages. "
            "The RAGTruth benchmark provides human-annotated span-level labels identifying "
            "hallucinated spans and their types (entity error, relation error, invented content) "
            "across QA, summarisation, and data-to-text tasks."
        ),
    },
    {
        "id": "rt_qa_017", "task": "qa", "split": "test",
        "question": "How does selective prediction help in question answering?",
        "source_info": (
            "Selective prediction allows a question answering system to abstain from answering "
            "when it is uncertain, trading coverage for accuracy. The theoretical framework of "
            "selective prediction (El-Yaniv and Wiener, 2010) formalises this as the "
            "coverage-risk tradeoff: a system with lower coverage achieves lower risk on answered "
            "questions. Practically, this means a system that abstains on 10% of questions and "
            "answers the remaining 90% can achieve significantly higher accuracy on its answered "
            "subset. Calibrated confidence scores are essential for effective selective prediction: "
            "the system should abstain precisely on the questions it is most likely to answer "
            "incorrectly."
        ),
    },
    {
        "id": "rt_qa_018", "task": "qa", "split": "test",
        "question": "What is domain adaptation in NLP?",
        "source_info": (
            "Domain adaptation in NLP refers to the process of adapting a model trained on "
            "general-domain data to perform well on a specific target domain, such as medicine, "
            "law, or finance. The domain shift problem arises when the statistical distribution "
            "of the target domain differs from the source (training) domain. Adaptation strategies "
            "include: (1) fine-tuning on domain-specific labelled data; (2) continued pre-training "
            "on domain-specific unlabelled text; (3) retrieval-based adaptation, where a "
            "domain-specific knowledge base is used without modifying model parameters; and "
            "(4) prompt-based adaptation using domain-specific exemplars. RAG is particularly "
            "well-suited for domain adaptation because the knowledge base can be updated without "
            "retraining the model."
        ),
    },
    {
        "id": "rt_qa_019", "task": "qa", "split": "test",
        "question": "What is the RAGTruth benchmark?",
        "source_info": (
            "RAGTruth is a hallucination benchmark for Retrieval-Augmented Generation systems, "
            "introduced by Wu et al. (2023). It consists of approximately 18,000 LLM-generated "
            "responses across three tasks: question answering (using MS MARCO passages), "
            "summarisation (using news articles), and data-to-text generation (using tables). "
            "Each response is annotated at the word or phrase level by human annotators who "
            "identify hallucinated spans and classify them into types including entity errors, "
            "relation errors, contradictions, and invented content. RAGTruth enables fine-grained "
            "evaluation of hallucination detection models and provides a standardised benchmark "
            "for comparing RAG systems in terms of faithfulness to retrieved context."
        ),
    },
    {
        "id": "rt_qa_020", "task": "qa", "split": "test",
        "question": "How is cosine similarity computed between embeddings?",
        "source_info": (
            "Cosine similarity measures the cosine of the angle between two vectors in a "
            "high-dimensional embedding space. For two vectors u and v, the cosine similarity "
            "is defined as their dot product divided by the product of their L2 norms: "
            "cos(u, v) = (u · v) / (||u|| ||v||). Cosine similarity ranges from -1 (opposite "
            "directions) to +1 (identical directions), with 0 indicating orthogonality. In "
            "retrieval systems, document embeddings are typically L2-normalised before indexing, "
            "after which inner product search is equivalent to cosine similarity search. Cosine "
            "similarity is preferred over Euclidean distance for high-dimensional embeddings "
            "because it is invariant to vector magnitude."
        ),
    },
    {
        "id": "rt_qa_021", "task": "qa", "split": "test",
        "question": "What is beam search in text generation?",
        "source_info": (
            "Beam search is a heuristic search algorithm used in sequence generation tasks to "
            "find high-probability output sequences. At each decoding step, beam search maintains "
            "a fixed number of candidate sequences (the beam width), expanding each candidate "
            "by all possible next tokens and keeping only the top-k candidates by cumulative "
            "log-probability. Unlike greedy decoding (which keeps only the single best token "
            "at each step), beam search explores multiple hypotheses simultaneously and produces "
            "globally more coherent sequences. A beam width of 4-5 is commonly used for "
            "abstractive summarisation and translation tasks. Larger beam widths improve quality "
            "but increase computational cost linearly."
        ),
    },
    {
        "id": "rt_qa_022", "task": "qa", "split": "test",
        "question": "What is the ROUGE metric?",
        "source_info": (
            "ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of automatic "
            "metrics for evaluating text summarisation and other NLG tasks by comparing generated "
            "text against one or more reference texts. ROUGE-N measures n-gram recall between "
            "the generated text and reference; ROUGE-1 uses unigrams, ROUGE-2 uses bigrams. "
            "ROUGE-L measures the longest common subsequence (LCS). ROUGE-1 F1, combining "
            "precision and recall, is the most commonly reported variant. ROUGE scores correlate "
            "moderately with human judgements for summarisation but are limited by their reliance "
            "on surface-level lexical overlap, failing to capture semantic equivalence between "
            "paraphrased content."
        ),
    },
    {
        "id": "rt_qa_023", "task": "qa", "split": "test",
        "question": "What is singular value decomposition (SVD)?",
        "source_info": (
            "Singular Value Decomposition (SVD) is a matrix factorisation technique that "
            "decomposes any real matrix M into three matrices: M = U Σ V^T, where U and V are "
            "orthogonal matrices and Σ is a diagonal matrix of non-negative singular values in "
            "descending order. Truncated SVD retains only the top-k singular values and "
            "corresponding singular vectors, producing a low-rank approximation of M that "
            "captures the most important variance. In NLP, SVD is used for Latent Semantic "
            "Analysis (LSA) to reduce TF-IDF matrices to dense low-dimensional representations. "
            "It is also used in dimensionality reduction for embeddings when neural encoders "
            "are unavailable."
        ),
    },
    {
        "id": "rt_qa_024", "task": "qa", "split": "test",
        "question": "What is the difference between precision and recall?",
        "source_info": (
            "Precision and recall are two fundamental evaluation metrics in information retrieval "
            "and classification. Precision measures the fraction of retrieved or predicted "
            "positives that are actually positive: Precision = TP / (TP + FP). Recall measures "
            "the fraction of actual positives that are retrieved or predicted: "
            "Recall = TP / (TP + FN). The F1 score is the harmonic mean of precision and recall: "
            "F1 = 2 * Precision * Recall / (Precision + Recall). In retrieval, high precision "
            "means most retrieved documents are relevant, while high recall means most relevant "
            "documents are retrieved. There is typically a precision-recall tradeoff: increasing "
            "recall often decreases precision."
        ),
    },
    {
        "id": "rt_qa_025", "task": "qa", "split": "test",
        "question": "What are large language model parameters?",
        "source_info": (
            "Language model parameters are the learnable weights of a neural network, adjusted "
            "during training to minimise the prediction loss. Modern LLMs are characterised by "
            "their parameter count: GPT-3 has 175 billion parameters, PaLM has 540 billion, "
            "and GPT-4 is estimated to have over 1 trillion. Parameters are distributed across "
            "attention weight matrices (Q, K, V, O projections), feed-forward network weights, "
            "layer normalisation parameters, and embedding tables. Larger parameter counts "
            "generally enable more capable models, but inference cost scales roughly linearly "
            "with parameter count. Techniques such as quantisation, pruning, and mixture-of-experts "
            "reduce effective parameter costs during inference."
        ),
    },
    # Out-of-domain negatives (should trigger abstention) 
    {
        "id": "rt_ood_001", "task": "qa", "split": "test",
        "question": "What is the boiling point of water at sea level?",
        "source_info": (
            "This passage discusses the history of the Olympic Games. The ancient Olympic Games "
            "were held at Olympia, Greece, from the 8th century BC to the 4th century AD. The "
            "modern Olympic Games were revived in 1896 in Athens, Greece. The Olympics are "
            "held every four years and include both Summer and Winter Games."
        ),
    },
    {
        "id": "rt_ood_002", "task": "qa", "split": "test",
        "question": "Who wrote the novel Pride and Prejudice?",
        "source_info": (
            "This document describes the architecture of the Eiffel Tower. The Eiffel Tower "
            "is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was "
            "constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair."
        ),
    },
]




@dataclass
class Document:
    id:          str
    task:        str          
    source_info: str          
    question:    str          
    split:       str = "train"

    @property
    def text(self) -> str:
        """Primary retrieval text."""
        return self.source_info

    @property
    def title(self) -> str:
        """Short human-readable label."""
        if self.question:
            return self.question[:80]
        return f"[{self.task}] {self.source_info[:60]}…"

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "task": self.task,
            "source_info": self.source_info,
            "question": self.question, "split": self.split,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Document":
        return cls(
            id=d.get("id", ""),
            task=d.get("task", "qa"),
            source_info=d.get("source_info", d.get("text", "")),
            question=d.get("question", ""),
            split=d.get("split", "train"),
        )



class RAGTruthCorpusLoader:
    

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def load(self) -> List[Document]:
        src = self.cfg.corpus_source
        if src == "ragtruth_builtin":
            docs = self._load_builtin()
        elif src == "ragtruth_json":
            docs = self._load_ragtruth_json()
        elif src == "json_file":
            docs = self._load_generic_json()
        else:
            raise ValueError(f"Unknown corpus_source: {src!r}")

        # Task filter
        if self.cfg.ragtruth_task_filter != "all":
            docs = [d for d in docs if d.task == self.cfg.ragtruth_task_filter]

        docs = [self._preprocess(d) for d in docs]
        docs = docs[: self.cfg.max_documents]
        logger.info(f"Corpus loaded: {len(docs)} documents "
                    f"(source={src}, task={self.cfg.ragtruth_task_filter})")
        return docs


    def _load_builtin(self) -> List[Document]:
        return [Document.from_dict(d) for d in RAGTRUTH_SAMPLE]

    def _load_ragtruth_json(self) -> List[Document]:
        path = self.cfg.corpus_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"RAGTruth JSON not found at {path!r}.\n"
                "Download with:\n"
                "  from datasets import load_dataset\n"
                "  ds = load_dataset('wandb/RAGTruth', split='train')\n"
                "  ds.to_json('data/ragtruth_train.jsonl')"
            )
        docs = []
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Map RAGTruth fields to our schema
                doc = Document(
                    id=d.get("id", str(i)),
                    task=d.get("task_type", d.get("task", "qa")).lower(),
                    source_info=d.get("source_info", d.get("passage", "")),
                    question=d.get("question", ""),
                    split=d.get("split", "train"),
                )
                docs.append(doc)
        return docs

    def _load_generic_json(self) -> List[Document]:
        path = self.cfg.corpus_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"JSON not found: {path!r}")
        with open(path) as f:
            raw = json.load(f) if path.endswith(".json") else [json.loads(l) for l in f if l.strip()]
        return [Document.from_dict(d) for d in raw]


    @staticmethod
    def _preprocess(doc: Document) -> Document:
        text = re.sub(r"\s+", " ", doc.source_info).strip()
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        doc.source_info = text
        return doc
