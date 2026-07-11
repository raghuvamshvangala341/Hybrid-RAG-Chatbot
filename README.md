# 🤖 Hybrid RAG Chatbot with Pinecone, BM25, Groq & Streamlit

A Retrieval-Augmented Generation (RAG) chatbot built using **Hybrid Search (Dense + Sparse Retrieval)**. The application enables users to upload documents, retrieve the most relevant information using Pinecone and BM25, and generate context-aware answers using the Groq LLM.

---

## 🚀 Features

* 📄 Upload PDF, DOCX, and TXT documents
* ✂️ Automatic document chunking
* 🧠 BGE embedding generation
* 🗂️ Pinecone vector database
* 🔍 Hybrid Search

  * Dense Retrieval (Pinecone)
  * Sparse Retrieval (BM25)
* 📌 Context-aware answer generation using Groq
* 📊 Runtime metrics dashboard
* 📑 Source citations with page numbers
* 💬 Streamlit chat interface
* ⚡ Modular architecture

---

## 🏗️ Project Architecture

```
                User Query
                     │
                     ▼
             Hybrid Retriever
        ┌────────────┴────────────┐
        ▼                         ▼
 Dense Search               BM25 Search
 (Pinecone)                 (Sparse)
        └────────────┬────────────┘
                     ▼
              Hybrid Score Fusion
                     ▼
           Context Construction
                     ▼
               Groq LLM
                     ▼
              Generated Answer
```

---

## 📂 Project Structure

```
rag_chatbot/
│
├── app.py
├── config.py
├── requirements.txt
├── .env
│
├── ingestion/
│   ├── loader.py
│   ├── splitter.py
│   ├── embedder.py
│   └── ingest.py
│
├── retrieval/
│   ├── bm25.py
│   ├── hybrid.py
│   └── retriever.py
│
├── vectorstore/
│   └── pinecone_store.py
│
├── cache/
│   └── embedding_cache.py
│
├── llm/
│   ├── prompt.py
│   ├── groq_client.py
│   └── rag_chain.py
│
├── ui/
│   ├── sidebar.py
│   ├── chat.py
│   └── metrics_panel.py
│
├── tests/
│
└── data/
```

---

## 🛠️ Tech Stack

| Component            | Technology                 |
| -------------------- | -------------------------- |
| Frontend             | Streamlit                  |
| LLM                  | Groq                       |
| Embeddings           | BAAI/bge-small-en-v1.5     |
| Vector Database      | Pinecone                   |
| Sparse Retrieval     | BM25                       |
| Dense Retrieval      | Pinecone Similarity Search |
| Programming Language | Python                     |
| Framework            | LangChain                  |

---

## ⚙️ Retrieval Pipeline

1. Upload a document.
2. Load and parse the document.
3. Split the content into chunks.
4. Generate embeddings.
5. Store embeddings in Pinecone.
6. Build the BM25 sparse index.
7. Receive a user query.
8. Perform Dense Retrieval.
9. Perform Sparse Retrieval.
10. Fuse retrieval scores.
11. Build the context.
12. Generate the final response using Groq.

---

## 📊 Runtime Metrics

### Retrieval Metrics

* Retrieval Time
* Retrieved Chunks
* Dense Retrieval Score
* BM25 Score
* Hybrid Score
* Average Cosine Similarity
* Maximum Cosine Similarity

### LLM Metrics

* Response Time
* Prompt Tokens
* Completion Tokens
* Total Tokens

### Ingestion Metrics

* Documents Indexed
* Chunks Created
* Embedding Time
* Indexing Time
* Total Ingestion Time
* Embedding Cache Size

---

## 📥 Installation

Clone the repository:

```bash
git clone <repository-url>
cd rag_chatbot
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key

PINECONE_API_KEY=your_pinecone_api_key

PINECONE_INDEX_NAME=rag-chatbot

LLM_MODEL=llama-3.3-70b-versatile

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

CHUNK_SIZE=1000

CHUNK_OVERLAP=200

TOP_K=5

HYBRID_ALPHA=0.7
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🧪 Testing

Run individual components:

```bash
python -m tests.test_loader
python -m tests.test_splitter
python -m tests.test_embedder
python -m tests.test_ingestion
python -m tests.test_retriever
python -m tests.test_groq
python -m tests.test_rag
```

---

## 📈 Future Enhancements

* Streaming responses
* Groundedness score
* Context precision
* Context recall
* Answer support evaluation
* Multi-document collections
* Conversation memory
* Authentication
* Conversation export

---

## 👨‍💻 Author

**Raghuvamsh Vangala**

B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)

Interests:

* Artificial Intelligence
* Machine Learning
* Natural Language Processing
* Retrieval-Augmented Generation (RAG)
* Large Language Models
* Generative AI
