# 🚀 ATHENA – AI-Powered Knowledge Management Platform

ATHENA is an intelligent platform designed to transform academic course materials (PDFs) into a **structured, searchable, and AI-augmented knowledge base**.  
It combines data processing, NLP, and embedding models to power a **semantic chatbot** capable of answering questions directly from course content.

---

## 🧭 Project Overview

> “From PDFs to an intelligent chatbot — ATHENA automates the entire journey of knowledge structuring.”

### 🧩 Pipeline Summary
**PDF → CSV → Cleaning → Tokenization → Embeddings → Chatbot**

| Phase | Description | Output |
|-------|--------------|---------|
| **1. Data Collection** | PDF extraction, metadata creation, and chunking into reusable course segments | `ATHENA_raw.csv` |
| **2. Cleaning & NLP Processing** | Dual cleaning tracks (`text_for_ner`, `clean_text_model`), tokenization, POS tagging, NER | `ATHENA_phase3_clean_variants.csv` |
| **3. Representation & Embeddings** | TF-IDF vectorization, Word2Vec (CBOW & Skip-gram), SBERT embeddings | `ATHENA_tfidf_embeddings.npz`, `ATHENA_cbow_skipgram_prep.npz` |
| **4. Application Layer** | Integration into a RAG chatbot (SBERT + FAISS + Gradio UI) | `athena_chatbot_demo.py` |

---
<img width="1163" height="671" alt="image" src="https://github.com/user-attachments/assets/a0956c85-68f0-4f39-93f5-fbd4babfc5a0" />
<img width="800" height="624" alt="image" src="https://github.com/user-attachments/assets/e8980162-7bb0-4bc4-97fb-518410b0f6f4" />
<img width="1821" height="765" alt="image" src="https://github.com/user-attachments/assets/11f32b44-f8ad-4a54-a379-758d15d88a38" />


## 🧹 Data Cleaning Strategy

### 🔹 `text_for_ner` (light cleaning)
- Keeps punctuation and capitalization for **Named Entity Recognition**
- Removes minimal artifacts: line breaks, hyphenations, invisible characters

### 🔹 `clean_text_model` (moderate cleaning)
- Converts text to lowercase
- Removes URLs/emails, extra spaces
- Keeps math/programming symbols `{ } [ ] ( ) + - = / < > % $ _`
- Applies light stopword filtering (EN/FR)

> 📊 6,711 text segments cleaned across 4 subjects (EN+FR)

---

## 🧠 NLP & Embedding Models

| Model | Goal | Tool / Library | Metric (Mean Similarity) |
|--------|------|----------------|---------------------------|
| **TF-IDF** | Baseline lexical representation | `sklearn.feature_extraction.text` | — |
| **CBOW** | Predicts a word from its context | PyTorch | 0.086 |
| **Skip-gram** | Predicts context from a target word | PyTorch | **0.475 (best)** |
| **SBERT** | Pretrained transformer for sentence embeddings | `sentence-transformers` | 0.230 |

> ⚖️ Skip-gram produced the most semantically consistent vectors for ATHENA’s corpus.

---

## 💬 Chatbot Application (RAG Pipeline)

Built using:
- 🧠 **SBERT** → Sentence embeddings for semantic search  
- 🗂️ **FAISS** → Fast vector indexing and retrieval  
- 🎨 **Gradio** → User interface for interactive querying  

### Example:
> **User:** “What is backpropagation?”  
> **ATHENA:** retrieves top 3 semantically relevant segments from Deep Learning course notes.

<p align="center">
  <img src="assets/athena_chat_demo.png" width="700"/>
</p>

---

## 📊 Evaluation Summary

| Metric | CBOW | Skip-gram | SBERT |
|---------|------|------------|--------|
| Intra-course similarity | 0.112 | **0.498** | — |
| Inter-course similarity | 0.067 | **0.459** | — |
| Mean similarity (off-diagonal) | 0.086 | **0.475** | 0.230 |

> 🏆 **Skip-gram** achieved the best semantic coherence across course topics.

---

## ⚙️ Tech Stack

- **Languages:** Python, Markdown  
- **Libraries:** pandas, spaCy, NLTK, PyTorch, FAISS, Gradio, Sentence-Transformers, scikit-learn  
- **Environment:** Google Colab / Jupyter Notebook  
- **Dataset:** Academic PDFs (Deep Learning, Computer Vision, Blockchain, Linear Programming)

---

## 🧩 Repository Structure

ATHENA/
│
├── data/
│ ├── blockchain_chunks.csv
│ ├── computer_vision_chunks.csv
│ ├── deepLearning_chunks.csv
│ └── pl_chunks.csv
│
├── notebooks/
│ └── DataProcessingProjectAthena.ipynb


## 📫 Contact
For collaboration or inquiries:  
**Aya Mekni**  
📧 [aya.mekni@esprim.tn]  
🌐 [www.linkedin.com/in/aya-mekni]  
🔗 [https://github.com/ayamekni]

---

⭐ **If you find this project interesting, give it a star!**  
