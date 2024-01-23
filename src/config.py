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
    path: str = "data/pdf"
    qdrant_api_key = os.getenv("QDRANT_API")
    qdrant_url = os.getenv("CLUSTER_ENDPOINT")
    collection_name: str ="ChatPDF"
    vector_size = 384
    vector_distance = Distance.COSINE
    reranker_model = "BAAI/bge-reranker-base"
    rrf_k = 60
    rerank_top_k = 5
    redis_port = 19440
    redis_host = "show-hyperfine-bone-11053.db.redis.io"
    redis_username = os.getenv("REDIS_USERNAME")
    redis_password = os.getenv("REDIS_PASSWORD")
    response_cache_ttl = 3600
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm_model="llama-3.1-8b-instant"
    llm_temperature=0.1
    max_token = 1024
    qdrant_batch_size: int = 50
    vector_top_k=5
    top_k =5

config = Config()