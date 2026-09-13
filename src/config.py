from qdrant_client.models import Distance
from dotenv import load_dotenv
from dataclasses import dataclass
import os

load_dotenv()


@dataclass
class Config:
    embedding_model: str = "all-MiniLM-L6-v2"
    chunking_model: str = "all-MiniLM-L6-v2"

    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: int = 95

    path: str = "../data/pdf"

    # Local Qdrant
    qdrant_api_key = None
    qdrant_url = "http://localhost:6333"

    collection_name: str = "ChatPDF"

    vector_size = 384
    vector_distance = Distance.COSINE

    # Reranking
    reranker_model = "BAAI/bge-reranker-base"

    # Reciprocal Rank Fusion
    rrf_k = 60

    rerank_top_k = 5

    # Local Redis
    redis_port = 6379
    redis_host = "localhost"
    redis_username = None
    redis_password = None

    response_cache_ttl = 3600

    # Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm_model = "openai/gpt-oss-20b"
    llm_temperature = 0.1
    max_token = 1024

    qdrant_batch_size: int = 50

    vector_top_k = 5
    top_k = 5


config = Config()