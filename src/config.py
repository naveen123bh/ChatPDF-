from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


# =============================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env when running locally
load_dotenv(
    BASE_DIR / ".env"
)


# =============================================================
# CONFIGURATION
# =============================================================

@dataclass
class Config:

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------

    path: str = str(
        BASE_DIR / "data" / "pdf"
    )

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    embedding_model: str = (
        "all-MiniLM-L6-v2"
    )

    vector_size: int = 384

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    vector_top_k: int = 5

    top_k: int = 5

    # Reciprocal Rank Fusion
    rrf_k: int = 60

    # CrossEncoder reranking
    reranker_model: str = (
        "BAAI/bge-reranker-base"
    )

    rerank_top_k: int = 5

    # ---------------------------------------------------------
    # Groq LLM
    # ---------------------------------------------------------

    groq_api_key: str | None = (
        os.getenv("GROQ_API_KEY")
    )

    llm_model: str = (
        "openai/gpt-oss-20b"
    )

    llm_temperature: float = 0.1

    max_token: int = 1024


# =============================================================
# DEFAULT CONFIG
# =============================================================

config = Config()