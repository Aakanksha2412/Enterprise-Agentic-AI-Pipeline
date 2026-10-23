# Enterprise Agentic AI Pipeline with Automated Continuous Evaluation

[![CI/CD Ragas Quality Gate](https://github.com/Aakanksha2412/Enterprise-Agentic-AI-Pipeline/actions/workflows/eval.yml/badge.svg)](https://github.com/Aakanksha2412/Enterprise-Agentic-AI-Pipeline/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-v1.10.0-red.svg)](https://qdrant.tech/)
[![Neo4j](https://img.shields.io/badge/Neo4j-v5.0.0-blue.svg)](https://neo4j.com/)
[![Redis](https://img.shields.io/badge/Redis-alpine-red.svg)](https://redis.io/)

A production-grade, multi-modal Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, **Qdrant**, **Neo4j**, and **Redis**. Features real-time Server-Sent Events (SSE) streaming, cross-database ingestion (`.md`, `.txt`, `.pdf`), and automated CI/CD quality gates powered by **Ragas** evaluation metrics (`faithfulness`, `answer_relevancy`, `context_precision`).

---

## 🏗️ System Architecture

```
                                  +------------------------------------+
                                  |         FastAPI REST Server        |
                                  |   (/chat/stream - SSE Endpoint)    |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |   LangGraph Agentic State Machine  |
                                  |     (Intent Router & Re-ranker)    |
                                  +-----------------+------------------+
                                                    |
          +-----------------------------------------+-----------------------------------------+
          |                                         |                                         |
          v                                         v                                         v
+-------------------+                     +-------------------+                     +-------------------+
|  Qdrant (Vector)  |                     |  Neo4j (Graph)    |                     |   Redis (Cache)   |
| Semantic Search   |                     | Entity Topologies |                     | Response Caching  |
+-------------------+                     +-------------------+                     +-------------------+
```

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Server Engine** | **FastAPI** | Asynchronous HTTP server with SSE streaming endpoints. |
| **Agent Orchestration** | **LangGraph** | Finite state machine for query routing and tool execution. |
| **Vector Database** | **Qdrant** | Dense semantic vector similarity search (`text-embedding-3-small`). |
| **Knowledge Graph** | **Neo4j** | Graph database mapping document-to-chunk entity topologies. |
| **Cache Layer** | **Redis** | In-memory key-value cache for normalized query caching. |
| **Reranking Engine** | **Cohere Rerank** | Contextual reranking model (`rerank-english-v3.0`). |
| **Evaluation Engine** | **Ragas & Pytest** | Automated continuous evaluation quality gates in GitHub Actions. |
| **Containerization** | **Docker Compose** | Multi-container environment management. |

---

## 🚀 Quick Start & Local Setup

### 1. Prerequisites
* **Docker Desktop** installed and running.
* **Python 3.11+** installed.
* An active **OpenAI API Key** and **Cohere API Key**.

### 2. Environment Configuration
Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

QDRANT_HOST=localhost
QDRANT_PORT=6333

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Spin Up Service Containers
Launch the database infrastructure stack:

```bash
docker compose up -d
```

Verify service readiness:
```bash
docker compose ps
```

---

## 📥 Ingestion Engine

Populate the database stack with private corporate documents (`.md`, `.txt`, `.pdf`).

1. Place source files inside the `./data` directory:
   ```bash
   mkdir -p data
   # Add your markdown or text files into ./data/
   ```

2. Execute the ingestion script:
   ```bash
   python -m app.ingest
   ```

The script automatically:
* Scans `./data/` using UTF-8 encoding.
* Chunks text into 500-character windows using `RecursiveCharacterTextSplitter`.
* Embeds vectors into Qdrant (`enterprise_docs` collection).
* Constructs document-chunk node graphs in Neo4j.

---

## 🧪 Automated Benchmarking & Ragas Quality Gates

Run local accuracy benchmarks using Ragas:

```bash
python -m pytest tests/test_ragas_eval.py -s
```

### Metric Thresholds Enforced
* **Faithfulness ($\ge 0.70$):** Verifies that generated responses do not contain ungrounded claims or hallucinations.
* **Answer Relevancy ($\ge 0.70$):** Ensures output directly addresses the user's input prompt.

---

## ⚡ Running the API Server

Start the FastAPI application server locally:

```bash
uvicorn app.main:app --reload
```

* **Interactive API Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Root Health Check:** [http://localhost:8000/](http://localhost:8000/)

### Test Real-Time SSE Token Streaming

Run via `curl`:
```bash
curl -N -X POST "http://localhost:8000/chat/stream?question=wealth%20mitra"
```

---

## ⚙️ CI/CD Pipeline Integration

The project includes an automated GitHub Actions workflow (`.github/workflows/eval.yml`) that triggers on every push or pull request to `main`:

1. Boots ephemeral containers (`qdrant`, `neo4j`, `redis`).
2. Polls Neo4j TCP port `7687` until ready.
3. Ingests mock seed dataset into vector and graph databases.
4. Runs `pytest tests/test_ragas_eval.py` to enforce quality thresholds before deployment.

---

## 📂 Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── eval.yml          # GitHub Actions CI/CD pipeline
├── app/
│   ├── __init__.py
│   ├── agent_graph.py        # LangGraph state graph routing logic
│   ├── ingest.py             # Multi-format ingestion script
│   └── main.py               # FastAPI server with Redis & SSE
├── data/                     # Ingestion source files (.md, .txt, .pdf)
├── tests/
│   └── test_ragas_eval.py    # Ragas metric evaluation suite
├── docker-compose.yml        # Multi-container database stack definition
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
