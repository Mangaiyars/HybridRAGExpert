# HybridRAGExpert
Hybrid RAG application
# ⚡ Hybrid RAG Application (FAISS + Neo4j + Guardrails + DeepEval)

A modular, production-ready Hybrid Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, **FAISS**, **Neo4j Aura Knowledge Graph**, **Cross-Encoder Neural Reranking**, and **DeepEval** for automated performance evaluation.

---

## 🏗️ Architecture & Pipeline Flow

The application executes a structured multi-stage pipeline:
1. **PDF Extraction & Chunking**: Parses uploaded runtime PDF files and splits them into manageable text sections.
2. **Parallel Ingestion**:
   - **Vector Pipeline**: Generates embeddings and builds a local **FAISS** vector database.
   - **Knowledge Graph Pipeline**: Extracts key entities and relationships using OpenAI and populates a **Neo4j Aura** graph database.
3. **Guardrails**: Validates user queries for quality and length before processing.
4. **Hybrid Retrieval**: Combines semantic matches from FAISS and graph context from Neo4j.
5. **Neural Reranking**: Uses `sentence-transformers` (`ms-marco-MiniLM-L-6-v2`) Cross-Encoder to bubble up the most precise context chunks.
6. **Unified Generation**: Synthesizes a grounded response using OpenAI (`gpt-4o-mini`).
7. **Automated Evaluation**: Measures **Faithfulness** and **Answer Relevancy** using **DeepEval**.

---

## 📁 Project Structure

All files reside in a single flat directory:
* **`app.py`**: The main Streamlit frontend entry point.
* **`config.py`**: Securely loads environment variables via `python-dotenv`.
* **`ingestion.py`**: Handles PDF loading, chunking, vector embedding creation, and Neo4j graph population.
* **`retrieval.py`**: Manages guardrails, hybrid search, cross-encoder reranking, LLM answer synthesis, and DeepEval evaluation metrics.
* **`requirements.txt`**: Project dependency list.
* **`.env`**: Configuration credentials (API keys and database URIs).

---

## 🛠️ Prerequisites & Installation

### 1. Install Dependencies
Navigate to your project directory in your terminal and install the required packages:
```bash
pip install -r requirements.txt
