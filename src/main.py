
from config import Config
from ingestion import DocumentIngestion
from vector_store import VectorStore
from cache import RedisCache
from retrieval import HybridRetriever, RAGPipeline
import contextlib
import io


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

    print("\n==============================")
    print("          ChatPDF")
    print("==============================")
    print("Loading PDF...\n")

    # Hide internal startup logs
    with contextlib.redirect_stdout(io.StringIO()):
        chunks = ingest()
        pipeline = build_pipeline(chunks)

    print("Ready.")
    print("Ask anything about your PDF.")
    print("Type 'exit' to quit.\n")

    while True:

        try:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            print("\nAI is thinking...\n")

            # Hide internal RAG logs
            with contextlib.redirect_stdout(io.StringIO()):
                answer = pipeline.answer(query)

            print(answer)
            print("\n" + "-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

        except Exception as e:
            print(f"\nError: {e}\n")
