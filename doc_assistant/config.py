"""
Configuration: API keys, LangChain LLM, and LlamaIndex global Settings.

Concepts used (from class):
- python-dotenv for API keys                                        (Day 9/10)
- ChatGroq "openai/gpt-oss-20b" / ChatOllama                         (Day 10, 13-18)
- LlamaIndex Settings.embed_model / Settings.llm                     (Day 14 - rag_llamaindex)
- sentence-transformers/all-MiniLM-L6-v2 local embeddings           (Day 13/14)
"""
import os
import re
import warnings
from pathlib import Path

# Mac (Apple Silicon) safety flags: avoid OpenMP clash + tokenizer fork warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv  # noqa: E402

warnings.filterwarnings("ignore")
load_dotenv()
# Hugging Face token (free) -> faster, rate-limit-free download of the embedding model
if os.getenv("HUGGINGFACE_API_KEY") and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY")

try:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
except NameError:                       # running inside the notebook
    PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data" / "documents"        # put your PDFs / txt / md files here
PERSIST_DIR = PROJECT_DIR / "storage"                 # LlamaIndex saves the vector index here

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()        # groq | gemini (free keys) | ollama (free, local)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))


def get_llm(temperature: float = 0.1):
    """LangChain chat model - LangChain does ALL the LLM work in this project."""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)

    if LLM_PROVIDER == "gemini":
        # Google Gemini - free key from https://aistudio.google.com/apikey  (same class as Day 9/15/17 notebooks)
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError("GEMINI_API_KEY is missing. Add it to your .env file.")
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature, google_api_key=key, max_retries=3)

    from langchain_groq import ChatGroq
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is missing. Add it to your .env file.")
    kwargs = dict(model=GROQ_MODEL, temperature=temperature, max_retries=5)
    if "gpt-oss" in GROQ_MODEL:
        kwargs["reasoning_effort"] = "low"
    return ChatGroq(**kwargs)


def configure_llamaindex(embed_model=None):
    """
    LlamaIndex is used ONLY for indexing + retrieval:
      - embeddings: local HuggingFace MiniLM (free, ~90 MB, fast on M1 CPU)
      - llm: disabled (None) because answer generation is done by LangChain
    """
    from llama_index.core import Settings
    from llama_index.core.node_parser import SentenceSplitter

    if embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device="cpu")
    Settings.embed_model = embed_model
    Settings.llm = None
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return Settings


def strip_think(text) -> str:
    """Remove <think>...</think> blocks that some local reasoning models output."""
    text = text if isinstance(text, str) else getattr(text, "content", str(text))
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def provider_label() -> str:
    model = {"groq": GROQ_MODEL, "gemini": GEMINI_MODEL, "ollama": OLLAMA_MODEL}.get(LLM_PROVIDER, GROQ_MODEL)
    return f"{LLM_PROVIDER}:{model}"
