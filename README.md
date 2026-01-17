
# Vector Stores – Complete Guide (with Examples)

## What is a Vector Store?
A **vector store** is a system designed to **store and retrieve data represented as numerical vectors (embeddings)**.
These embeddings are generated from text, images, audio, or other unstructured data using embedding models.

Vector stores are a core building block of modern AI systems such as **semantic search** and **Retrieval-Augmented Generation (RAG)**.

---

## Key Features

### 1. Storage
- Stores vectors along with their associated **metadata**
- Storage options:
  - **In-memory (RAM)** for fast lookups
  - **On-disk (Hard disk)** for durability and large-scale usage

### 2. Similarity Search
- Retrieves vectors that are **most similar** to a query vector
- Common similarity metrics:
  - Cosine similarity
  - Euclidean distance
  - Dot product

### 3. Indexing
- Uses optimized data structures to enable **fast similarity search**

### 4. CRUD Operations
- **Create** – Add new vectors
- **Read** – Retrieve vectors
- **Update** – Modify vectors or metadata
- **Delete** – Remove outdated vectors

---

## Use Cases
- Semantic Search
- Retrieval-Augmented Generation (RAG)
- Recommendation Systems
- Image / Multimedia Search

---

## Vector Store vs Vector Database

### Vector Store
A **vector store** is a lightweight library or service focused on:
1. Storing embeddings
2. Performing similarity search

**Characteristics:**
- Minimal database features
- No transactions
- No role-based access control
- Ideal for prototyping and small-scale applications

**Example:** FAISS

---

### Vector Database
A **vector database** is a fully-fledged database system designed for production-scale workloads.

**Additional Features:**
- Distributed architecture (horizontal scaling)
- Durability and persistence (replication, backup, restore)
- Metadata handling with schema and filters
- ACID transactions
- Concurrency control
- Authentication and authorization

**Best suited for:**
- Large-scale production environments

**Examples:**
- Qdrant
- Milvus
- Pinecone

> Every vector database is a vector store, but not every vector store is a vector database.

---

## Vector Stores in LangChain

LangChain provides **wrappers** around popular vector stores with a **common interface**.

### Supported Vector Stores
- FAISS
- Chroma
- Pinecone
- Qdrant
- Weaviate

### Common APIs
```python
from_documents(...)
from_texts(...)

add_documents(...)
add_texts(...)

similarity_search(query, k=5)
```

### Metadata-Based Filtering
```python
similarity_search(
    query="Who is Rohit Sharma?",
    k=3,
    filter={"team": "Mumbai Indians"}
)
```

**Advantage:** You can switch vector stores with minimal code changes.

---

## ChromaDB

**Chroma** is a lightweight, open-source vector database suitable for:
- Local development
- Learning and experimentation
- Small to medium-scale production systems

---

## ChromaDB Hierarchy

```
Tenant
 └── Database
      └── Collection
           └── Document
               ├── Embedding Vector
               └── Metadata
```

Each document contains:
- An embedding vector
- Metadata (tags, source, filters, etc.)

---

## Example: Chroma with LangChain

### Installation
```bash
pip install langchain-chroma chromadb sentence-transformers
```

### Code Example
```python
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

documents = [
    Document(page_content="Rohit Sharma is an Indian cricketer", metadata={"team": "Mumbai Indians"}),
    Document(page_content="Virat Kohli is an Indian cricketer", metadata={"team": "RCB"})
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

results = vectorstore.similarity_search(
    query="Who plays for Mumbai Indians?",
    k=2,
    filter={"team": "Mumbai Indians"}
)

for r in results:
    print(r.page_content)
```

---

## Get All Document IDs
```python
data = vectorstore.get(include=["ids"])
print(data["ids"])
```

---

## Summary
- Vector stores enable semantic search using embeddings
- Vector databases add scalability and reliability
- LangChain provides a unified interface
- Chroma is ideal for local and small-scale production use
