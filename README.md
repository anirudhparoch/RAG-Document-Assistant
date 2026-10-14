# Assignment 1.2 – RAG-Based Intelligent Document Assistant (LangChain + LlamaIndex)

A document question-answering system: **LlamaIndex** ingests, indexes and retrieves; **LangChain** orchestrates the LLM, prompts, memory and output parsing. Answers include numbered citations, a confidence level and follow-up suggestions. It can also summarise a document and compare several documents.

## Deliverables
| Deliverable | File |
|---|---|
| Jupyter Notebook | `RAG_Document_Assistant.ipynb` |
| Deployable application | `app.py` (Streamlit) + `doc_assistant/` package |
| Presentation | `presentation/RAG_Document_Assistant.pptx` (+ PDF) |

## Architecture
```
 PDF / TXT / MD / URL
        │  SimpleDirectoryReader / BeautifulSoup        ┐
        ▼                                               │
 SentenceSplitter (512 tokens, 64 overlap)              │  LlamaIndex
        ▼                                               │
 HuggingFace all-MiniLM-L6-v2 embeddings                │
        ▼                                               │
 VectorStoreIndex ──persist──► ./storage                │
        ▼                                               │
 VectorIndexRetriever (+ MetadataFilters per document)  │
        ▼                                               │
 SimilarityPostprocessor (cutoff)                       ┘
        ▼
 RunnableLambda(retriever) → RunnableParallel → ChatPromptTemplate → ChatGroq → PydanticOutputParser   ┐ LangChain
 + condense-question chain (memory) · map-reduce summary chain · multi-document compare chain          ┘
```

## Features
- Ingest PDFs, text, markdown and **web pages**; add new files without rebuilding the index
- Persistent index (`./storage`), so restarts are fast
- Semantic retrieval with adjustable top-k and similarity cutoff
- **Context-aware chat**: follow-up questions are rewritten into standalone questions
- **Structured answers** (`RAGAnswer`: answer, sources used, confidence, follow-up questions)
- **Multi-document** querying (filter by document), **summarisation** and **comparison**
- Refuses to answer when the information is not in the documents

## Setup on Mac (M1, 16 GB RAM)
1. Open this folder in VS Code: **File → Open Folder… → Assignment_1_2_RAG_Document_Assistant**
2. In the terminal (**Ctrl + `**):
   ```bash
   bash setup_mac.sh
   ```
   Creates `.venv`, installs `requirements.txt` and creates `.env`. Needs Python 3.11 or 3.12 (`brew install python@3.12` or `uv python install 3.12`).
3. Paste your free Groq key (<https://console.groq.com/keys>) into `.env` → `GROQ_API_KEY=gsk_...`
4. **Notebook:** open `RAG_Document_Assistant.ipynb` → *Select Kernel* → `.venv` → **Run All**.
5. **App:**
   ```bash
   source .venv/bin/activate
   streamlit run app.py
   ```
   or press **F5** in VS Code. Opens at <http://localhost:8501>.

The first run downloads the embedding model (~90 MB) and indexes the 3 sample PDFs (~280 chunks, under a minute on an M1). After that the index loads from `./storage`.
To re-index from scratch, delete the `storage/` folder.

### Fully offline option
Install [Ollama](https://ollama.com) → `ollama pull qwen3.5:2b` → set `LLM_PROVIDER=ollama` in `.env`.


### Free API keys used (all set in `.env`)
| Service | Used for | Free key |
|---|---|---|
| Groq (`openai/gpt-oss-120b`) | Main LLM (`LLM_PROVIDER=groq`) | https://console.groq.com/keys |
| Google Gemini (`gemini-3.6-flash`) | Backup LLM: set `LLM_PROVIDER=gemini` if Groq hits its daily limit | https://aistudio.google.com/apikey |
| Hugging Face | Faster download of the local embedding model | https://huggingface.co/settings/tokens |
| Tavily (Stock Assistant only) | News search | https://app.tavily.com |
No paid service (e.g. OpenAI) is used anywhere.

## Deploy (Streamlit Community Cloud)
Push to GitHub (`.env` and `storage/` are git-ignored) → <https://share.streamlit.io> → *New app* → `app.py` → *Secrets*:
```toml
GROQ_API_KEY = "gsk_..."
GEMINI_API_KEY = "..."
HUGGINGFACE_API_KEY = "hf_..."
```

## Project structure
```
├── RAG_Document_Assistant.ipynb   # step-by-step notebook
├── app.py                         # Streamlit app
├── doc_assistant/
│   ├── config.py      # .env, LangChain LLM, LlamaIndex Settings
│   ├── indexer.py     # LlamaIndex ingestion, index, persistence, retrieval
│   └── rag_chain.py   # LangChain chains: QA, memory, summary, compare
├── data/documents/    # sample PDFs from class (add your own here)
├── presentation/
├── requirements.txt · .env.example · setup_mac.sh
```

## Sample questions
- *What is ReAct and how is it different from chain-of-thought?*
- *Which benchmarks did they evaluate it on?* (follow-up)
- *Explain scaled dot-product attention.*
- *What are the main causes of hallucination in LLMs?*
- Compare tab: *What problem does each paper solve?*
