# NOVEC - Vector-less Retrieval-Augmented Generation (RAG)

## Overview

NOVEC is a vector-less RAG system that leverages PageIndex's hierarchical document structure and OpenAI's language models to answer questions based on uploaded documents. 

Instead of using vector embeddings, NOVEC utilises intelligent tree-based searching and LLM reasoning to identify and extract relevant content from documents.


---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Document Processing** | PageIndex API |
| **LLM (Queries & Analysis)** | OpenAI GPT (e.g., gpt-5-nano) |
| **CLI Interface** | Rich library |
| **Web UI** | Streamlit |
| **API Communication** | Python requests |
| **Python** | 3.8+ |

---

## Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# API Keys and Services
# - PageIndex account with API key
# - OpenAI account with API key
```

### Configuration (.env)

Create a `.env` file in the project root:

```bash
# PageIndex API
PAGEINDEX_API_KEY=your_pageindex_api_key_here

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: for custom endpoints

# Model Configuration
MODEL_NAME=gpt-5-nano  # Or your preferred GPT model
```

### Run CLI Interface

```bash
python -m applications/cli_app.py
```

**Menu Options:**
```
NOVEC RAG (CLI)
Choose an option:
  1. Write a query
  2. Upload a document
  3. Delete a document
  4. Exit
```

### Run Streamlit Web UI

```bash
python -m streamlit run applications/streamlit_app.py
```

**Access:** Open browser to `http://localhost:8501`

---

## Project Structure

```
novec-vectorless-RAG/
├── novec.py                          # Core backend logic
├── config.py                         # Configuration management
├── cli_utils.py                      # CLI utility functions
│
├── applications/
│   ├── cli_app.py                   # CLI application
│   └── streamlit_app.py             # Streamlit web UI
│
├── requirements.txt                 # Python dependencies
├── .env                            # Configuration (create this)
└── README.md                        # This file
```

---

## Component Documentation

### Core Modules

| Module | Purpose |
|--------|---------|
| `novec.py` | **Main backend**: RAGConfig, PageIndexAPI, RAGEngine, QueryExecutor |
| `config.py` | Configuration loading and management |
| `cli_utils.py` | CLI validation and utility functions |

### PageIndex Integration (`PageIndexAPI`)

| Method | Purpose |
|--------|---------|
| `upload_document(file_path)` | Upload PDF and return document ID |
| `wait_for_indexing(doc_id)` | Poll until document is indexed and ready |
| `fetch_documents(limit, offset)` | Get list of uploaded documents |
| `delete_document(doc_id)` | Remove document from system |
| `get_tree(doc_id)` | Fetch hierarchical document structure |
| `get_document_metadata(doc_id)` | Get status, page count, timestamps |

### RAG Operations (`RAGEngine`)

| Method | Purpose |
|--------|---------|
| `extract_document_structure(tree)` | Convert tree to readable outline |
| `identify_relevant_sections(query, structure, doc_name)` | Use LLM to find query-relevant sections |
| `llm_tree_search(query, tree)` | LLM-based tree node selection |
| `find_nodes_by_ids(tree, target_ids)` | Retrieve specific content nodes |
| `extract_full_content(tree)` | Flatten tree into all content nodes |
| `extract_asset_references(nodes)` | Identify figures, tables, diagrams |
| `generate_answer(query, nodes)` | Single-document answer generation |
| `generate_cited_answer(query, context_by_doc)` | Multi-document answer with citations |

### Query Orchestration (`QueryExecutor`)

Implements 3-step query pipeline:
1. **Verify Documents** - Confirm documents are indexed and ready
2. **Extract Content** - Apply size-based strategy (full vs. selective)
3. **Generate Answer** - Create cited response using OpenAI

```python
execute_query(query, selected_docs, progress_callback)
  → Returns: {success, answer, citations, steps_log, error}
```

---
