# 📄 ChatPDF — Advanced Hybrid RAG Pipeline

> An enterprise-grade Retrieval-Augmented Generation (RAG) system that lets you chat with your PDF documents using hybrid search, semantic reranking, and intelligent caching.

---

## 🧠 What is ChatPDF?

ChatPDF is a production-ready RAG pipeline that ingests PDF documents, stores them as semantic vectors in a cloud vector database, and answers natural language questions by retrieving the most relevant context using a combination of keyword and semantic search — then generating precise answers via a large language model.

Unlike basic RAG systems that rely solely on vector similarity, ChatPDF uses **Hybrid Retrieval** (BM25 + Vector Search) fused with **Reciprocal Rank Fusion (RRF)** and **CrossEncoder Reranking** to deliver highly accurate, grounded answers.

---

## 🏭 Industry Use Cases

| Domain | Use Case |
|---|---|
| **Legal** | Query contracts, case files, and compliance documents instantly |
| **Healthcare** | Search through medical reports, research papers, and clinical notes |
| **Finance** | Extract insights from annual reports, filings, and prospectuses |
| **Education** | Build intelligent tutoring systems over textbooks and lecture notes |
| **Enterprise** | Internal knowledge base search over HR policies, SOPs, and manuals |
| **Research** | Semantic search across large collections of academic papers |

---

## ⚙️ Tech Stack

| Technology | Role |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core language |
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) | REST API layer |
| ![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat&logo=redis&logoColor=white) | Embedding + response caching |
| ![Qdrant](https://img.shields.io/badge/Qdrant-FF4785?style=flat&logo=qdrant&logoColor=white) | Vector database |
| ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white) | Document loading, chunking orchestration |
| ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black) | Embedding + reranker models |
| ![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logo=groq&logoColor=white) | LLM inference (LLaMA 3.1) |

---

## 🏗️ Project Structure

```
ChatPDF/
├── src/
│   ├── config.py          # All settings — models, paths, API keys
│   ├── cache.py           # Redis embedding + response cache
│   ├── ingestion.py       # PDF loading + semantic chunking
│   ├── vector_store.py    # Qdrant vector DB operations
│   ├── retrieval.py       # BM25 + vector hybrid retrieval, RRF, reranking, RAG pipeline
│   └── main.py            # Entry point — wires everything together
├── data/
│   └── pdf/               # Drop your PDF files here
├── .env                   # API keys (never commit this)
├── requirements.txt
└── README.md
```

---

## 🔄 System Workflow

```
                        ┌─────────────────────────────────────┐
                        │           INGESTION PIPELINE         │
                        └─────────────────────────────────────┘

  PDF Files
      │
      ▼
  PyMuPDF Loader          ← Loads raw text + metadata (source, page number)
      │
      ▼
  Semantic Chunker         ← Splits text by meaning, not fixed size
  (HuggingFace Embeddings  (breakpoint_threshold_type = percentile @ 95)
   all-MiniLM-L6-v2)
      │
      ▼
  SentenceTransformer      ← Encodes each chunk → 384-dim float vector
  (all-MiniLM-L6-v2)           ↕ Redis checks cache first
      │
      ▼
  Qdrant Upsert            ← Stores vectors + payload (text, source, page)
  (batched, 50 at a time)      Skips duplicates via MD5 hash ID


                        ┌─────────────────────────────────────┐
                        │            QUERY PIPELINE            │
                        └─────────────────────────────────────┘

  User Query
      │
      ▼
  Redis Response Cache     ← Return immediately if query seen before
      │ (cache miss)
      ▼
  Redis Embedding Cache    ← Return cached vector or encode + store
      │
      ├─────────────────────────────────────┐
      ▼                                     ▼
  BM25 Retriever                     Qdrant Vector Search
  (keyword match, top 10)            (cosine similarity, top 10)
      │                                     │
      └──────────────┬──────────────────────┘
                     ▼
              RRF Fusion               ← Merges both ranked lists
              (k=60)                       score = Σ 1/(k + rank)
                     │
                     ▼
              CrossEncoder Reranker    ← Scores (query, chunk) pairs
              (BAAI/bge-reranker-base)     Keeps top 5
                     │
                     ▼
              Prompt Builder           ← Injects context into prompt
                     │
                     ▼
              Groq LLM                 ← LLaMA 3.1 8B generates answer
              (llama-3.1-8b-instant)
                     │
                     ▼
              Redis Cache Response     ← Store for 1 hour (TTL)
                     │
                     ▼
              Final Answer ✅
```

---

## 🧮 Algorithms Explained

### 1. Semantic Chunking
Unlike fixed-size chunking (e.g. every 500 tokens), **SemanticChunker** from LangChain embeds sentences and splits at points where the embedding similarity drops below a percentile threshold (95th percentile here). This ensures each chunk is semantically coherent — a chunk won't cut mid-idea.

### 2. Dense Embedding — `all-MiniLM-L6-v2`
- Architecture: MiniLM (distilled BERT), 6 transformer layers
- Output: 384-dimensional float vector
- Trained on: 1B+ sentence pairs for semantic similarity
- Why this model: fast, lightweight, strong semantic performance for its size

### 3. BM25 (Sparse Retrieval)
Best Match 25 is a probabilistic keyword ranking algorithm. It scores documents by term frequency (TF) and inverse document frequency (IDF), with length normalization. It excels at **exact keyword matches** where dense embeddings may miss (e.g. proper nouns, model names, acronyms).

```
score(D, Q) = Σ IDF(qi) × [tf(qi,D) × (k1+1)] / [tf(qi,D) + k1×(1 - b + b×|D|/avgdl)]
```

### 4. Reciprocal Rank Fusion (RRF)
RRF merges two ranked lists (BM25 + vector) into a single fused ranking without needing score normalization. Each document gets a score based on its rank position in each list:

```
RRF_score(doc) = Σ 1 / (k + rank(doc))     where k = 60
```

Documents appearing high in both lists get boosted. Documents missing from one list still contribute via the other.

### 5. CrossEncoder Reranking — `BAAI/bge-reranker-base`
After RRF, a CrossEncoder scores each `(query, chunk)` pair jointly — unlike bi-encoders which embed them separately. This is slower but far more accurate for relevance scoring because the model can directly attend across both texts.

```
Input:  [CLS] query [SEP] chunk [SEP]
Output: scalar relevance score
```

Top 5 scoring chunks are kept and passed to the LLM.

### 6. Redis Embedding Cache
Every text → vector conversion is cached in Redis using an MD5 hash of the text as the key, stored as raw `float32` bytes. On repeated queries or re-ingestion, embeddings are served from cache in microseconds instead of running the model.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Redis running locally (`localhost:6379`)
- Qdrant cloud account — [cloud.qdrant.io](https://cloud.qdrant.io)
- Groq API key — [console.groq.com](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ChatPDF.git
cd ChatPDF
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the root directory:
```env
CLUSTER_ENDPOINT=https://your-cluster-url.qdrant.io
QDRANT_API=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

### 5. Start Redis
```bash
# Using Docker (recommended)
docker run -d -p 6379:6379 redis

# Or if installed locally
redis-server
```

### 6. Add your PDFs
```bash
# Drop your PDF files into:
data/pdf/your_document.pdf
```

### 7. Run the pipeline
```bash
python src/main.py
```

---

## 🔧 Configuration

All settings are in `src/config.py`. Key parameters you may want to tune:

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `all-MiniLM-L6-v2` | Model for generating vectors |
| `reranker_model` | `BAAI/bge-reranker-base` | CrossEncoder reranker |
| `chunk_threshold_amount` | `95.0` | Semantic chunking sensitivity |
| `bm25_top_k` | `10` | BM25 candidates to retrieve |
| `vector_top_k` | `10` | Vector search candidates |
| `rerank_top_k` | `5` | Final chunks after reranking |
| `rrf_k` | `60` | RRF constant (higher = less rank bias) |
| `qdrant_batch_size` | `50` | Points per upsert batch |
| `response_cache_ttl` | `3600` | Redis response TTL in seconds |
| `llm_model` | `llama-3.1-8b-instant` | Groq LLM model |

---

## 📦 Requirements

```txt
langchain
langchain-community
langchain-experimental
langchain-huggingface
langchain-groq
qdrant-client
sentence-transformers
redis
python-dotenv
pymupdf
rank-bm25
fastapi
uvicorn
numpy
```

---

## 🧩 Class Responsibilities

| Class | File | Responsibility |
|---|---|---|
| `Config` | `config.py` | Single source of truth for all settings |
| `RedisCache` | `cache.py` | Embedding cache + response cache via Redis |
| `DocumentIngestion` | `ingestion.py` | Load PDFs, semantic chunking |
| `VectorStore` | `vector_store.py` | Qdrant collection management, ingest, search |
| `HybridRetriever` | `retrieval.py` | BM25 + vector search, RRF fusion, reranking |
| `RAGPipeline` | `retrieval.py` | Orchestrates retrieval → prompt → LLM → cache |

---

## ⚡ Performance Optimisations

- **Batch upserts** — vectors are sent to Qdrant in batches of 50 to avoid write timeouts
- **Batch existence check** — all chunk IDs checked in one Qdrant `retrieve()` call instead of N calls
- **Redis embedding cache** — avoids re-encoding identical text across runs
- **Redis response cache** — identical queries served in milliseconds (1 hour TTL)
- **Lazy model loading** — HuggingFace chunking model only loaded when first needed

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

> Built with ❤️ using Python, LangChain, Qdrant, Redis, and Groq.
