"""
LlamaIndex part: document ingestion, indexing, persistence and retrieval.

Concepts used (from class - Day 14 rag_llamaindex.ipynb):
- SimpleDirectoryReader                 -> load PDFs / text files
- VectorStoreIndex.from_documents       -> chunk + embed + index
- StorageContext / load_index_from_storage -> persist index to ./storage and reload
- VectorIndexRetriever                  -> top-k semantic retrieval
- SimilarityPostprocessor               -> drop chunks below a similarity cutoff
"""
import re
from pathlib import Path

import requests
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from .config import DATA_DIR, PERSIST_DIR

SUPPORTED_EXTS = [".pdf", ".txt", ".md"]


# ------------------------------------------------------------------ Ingestion
def load_documents_from_folder(folder=DATA_DIR) -> list:
    folder = Path(folder)
    if not folder.exists() or not any(p.suffix.lower() in SUPPORTED_EXTS for p in folder.iterdir()):
        return []
    reader = SimpleDirectoryReader(input_dir=str(folder), required_exts=SUPPORTED_EXTS, recursive=True)
    return reader.load_data(show_progress=True)


def load_files(paths: list) -> list:
    return SimpleDirectoryReader(input_files=[str(p) for p in paths]).load_data()


def load_web_page(url: str) -> list:
    """Fetch a web page and turn it into a LlamaIndex Document (web data ingestion)."""
    from bs4 import BeautifulSoup

    html = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    title = (soup.title.string or url).strip() if soup.title else url
    text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n")).strip()
    name = re.sub(r"[^A-Za-z0-9]+", "_", title)[:60] + ".web"
    return [Document(text=text, metadata={"file_name": name, "source_url": url, "title": title})]


# ------------------------------------------------------------------ Index
def build_index(documents: list, persist_dir=PERSIST_DIR) -> VectorStoreIndex:
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    return index


def load_or_build_index(data_dir=DATA_DIR, persist_dir=PERSIST_DIR, rebuild: bool = False):
    """Load the saved index if it exists, otherwise build it from the documents folder."""
    persist_dir = Path(persist_dir)
    if persist_dir.exists() and any(persist_dir.iterdir()) and not rebuild:
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        return load_index_from_storage(storage_context)
    docs = load_documents_from_folder(data_dir)
    if not docs:
        index = VectorStoreIndex.from_documents([])      # empty index; documents can be added later
        index.storage_context.persist(persist_dir=str(persist_dir))
        return index
    return build_index(docs, persist_dir)


def add_documents(index: VectorStoreIndex, documents: list, persist_dir=PERSIST_DIR) -> int:
    """Insert new documents into an existing index and save it."""
    for d in documents:
        index.insert(d)
    index.storage_context.persist(persist_dir=str(persist_dir))
    return len(documents)


def list_documents(index: VectorStoreIndex) -> list:
    """Unique file names in the index (PDFs are stored one Document per page)."""
    names = {info.metadata.get("file_name", "unknown") for info in index.ref_doc_info.values()}
    return sorted(names)


def get_document_nodes(index: VectorStoreIndex, file_name: str) -> list:
    """All chunks (nodes) of one document, in reading order."""
    nodes = [n for n in index.docstore.docs.values() if n.metadata.get("file_name") == file_name]

    def order(n):
        page = str(n.metadata.get("page_label", "0"))
        return (int(page) if page.isdigit() else 0, n.start_char_idx or 0)

    return sorted(nodes, key=order)


# ------------------------------------------------------------------ Retrieval
def make_filters(file_names: list | None):
    if not file_names:
        return None
    return MetadataFilters(filters=[MetadataFilter(key="file_name", value=list(file_names), operator=FilterOperator.IN)])


def retrieve(index: VectorStoreIndex, question: str, top_k: int = 4, file_names: list | None = None,
             similarity_cutoff: float = 0.25) -> list:
    """Semantic retrieval with LlamaIndex: VectorIndexRetriever + SimilarityPostprocessor."""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k, filters=make_filters(file_names))
    nodes = retriever.retrieve(question)
    kept = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff).postprocess_nodes(nodes)
    return kept or nodes[:2]           # never return nothing if the index has data


def query_engine_preview(index: VectorStoreIndex, question: str, top_k: int = 4):
    """
    LlamaIndex query engine (as in class). With Settings.llm = None it runs in 'no_text' mode
    and returns only the retrieved source nodes - the answer itself is generated by LangChain.
    """
    engine = index.as_query_engine(similarity_top_k=top_k, response_mode="no_text")
    return engine.query(question)
