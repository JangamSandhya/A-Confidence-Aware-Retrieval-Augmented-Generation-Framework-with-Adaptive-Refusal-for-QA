# Confidence-Aware RAG: Adaptive Refusal for Hallucination Reduction

This repository contains the official implementation of the paper:

> **"A Confidence-Aware Retrieval-Augmented Generation Framework with Adaptive Refusal for Question Answering"**  
> *Jangam Sandhya, Ajay Nehra*  
> Indian Institute of Information Technology (IIIT) Kota, India

The system adds a lightweight confidence estimation step after retrieval to decide whether to **generate**, **hedge**, or **abstain** from answering. It prevents hallucinations caused by irrelevant or out‑of‑domain documents without any retraining or auxiliary models.

---

## ✨ Key Features

- **Four confidence methods**: Score Gap, Margin, Entropy, Combined ensemble  
- **Three‑way decision**: Generate (high confidence) / Hedge (medium) / Abstain (low)  
- **Training‑free**: Works with any retriever and generator  
- **Domain‑agnostic**: Tested on medical (PubMed) and AI/NLP (RAGTruth) corpora  
- **Zero extra compute**: Uses only existing retrieval similarity scores  
- **100% OOD abstention**: Perfect refusal on out‑of‑domain questions  

---

## 📊 Results Summary

| Corpus | Baseline ROUGE‑1 | Confidence‑Aware ROUGE‑1 (answered) | OOD Abstention | Coverage |
|--------|----------------|--------------------------------------|----------------|----------|
| PubMed (32 docs) | 0.5932 | 0.6353 (+7.10%) | 100% | 88.24% |
| RAGTruth (35 docs) | 0.2943 | 0.3536 (+20.15%) | 100% | 76.47% |

---

## 🚀 Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/confidence-aware-rag.git
cd confidence-aware-rag
