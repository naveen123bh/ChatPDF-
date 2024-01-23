from config import Config
from ingestion import DocumentIngestion
from vector_store import VectorStore
from cache import RedisCache
from retrieval import HybridRetriever, RAGPipeline

config = Config()

def build_pipeline(chunks) -> RAGPipeline:
    cache = RedisCache(config)
    store = VectorStore(config, cache)
    retriever = HybridRetriever(chunks, store, config)
    return RAGPipeline(config, retriever, cache, store)

def ingest():
    loader = DocumentIngestion(config)
    chunks = loader.load_and_chunk()

    cache = RedisCache(config)
    store = VectorStore(config, cache)
    store.ingest(chunks)
    return chunks


if __name__ == "__main__":
    chunks = ingest()
    pipeline = build_pipeline(chunks)

    query = "What are Transformers?"
    answer = pipeline.answer(query)
    print("\nFINAL ANSWER:\n", answer)