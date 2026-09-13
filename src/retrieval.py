from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from src.vector_store import VectorStore
from src.config import Config
from src.cache import RedisCache


class HybridRetriever:

    def __init__(
        self,
        chunks: list[Document],
        vectorStore: VectorStore,
        config: Config
    ):
        self.chunks = chunks
        self.vectorStore = vectorStore
        self.config = config

        # -----------------------------------------------------
        # BM25 lexical retrieval
        # -----------------------------------------------------

        self.bm25 = BM25Retriever.from_documents(
            chunks,
            k=self.config.vector_top_k
        )

        # -----------------------------------------------------
        # CrossEncoder reranker
        # -----------------------------------------------------

        self.reranker = CrossEncoder(
            config.reranker_model
        )

    # ---------------------------------------------------------
    # Reciprocal Rank Fusion
    # ---------------------------------------------------------

    def _rrf(
        self,
        bm25_docs: list[Document],
        vector_docs: list[Document]
    ) -> list[Document]:

        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        k = self.config.rrf_k

        # BM25 ranking
        for rank, doc in enumerate(bm25_docs):

            content = doc.page_content

            doc_map[content] = doc

            scores[content] = (
                scores.get(content, 0)
                + 1 / (k + rank + 1)
            )

        # Vector ranking
        for rank, doc in enumerate(vector_docs):

            content = doc.page_content

            doc_map[content] = doc

            scores[content] = (
                scores.get(content, 0)
                + 1 / (k + rank + 1)
            )

        # Highest RRF score first
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc_map[content]
            for content, _ in ranked
        ]

    # ---------------------------------------------------------
    # CrossEncoder reranking
    # ---------------------------------------------------------

    def _rerank(
        self,
        query: str,
        docs: list[Document]
    ) -> list[Document]:

        if not docs:
            return []

        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        scores = self.reranker.predict(
            pairs
        )

        scored = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        top = scored[
            :self.config.rerank_top_k
        ]

        for i, (doc, score) in enumerate(
            top,
            1
        ):

            print(
                f"Rank {i} | "
                f"Score: {score:.4f} | "
                f"{doc.page_content[:80]}..."
            )

        return [
            doc
            for doc, _ in top
        ]

    # ---------------------------------------------------------
    # Hybrid retrieval
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        query_vector: list[float]
    ) -> list[Document]:

        # 1. BM25 lexical search
        bm25_docs = self.bm25.invoke(
            query
        )

        # 2. FAISS semantic search
        vector_docs = self.vectorStore.search(
            query_vector,
            self.config.vector_top_k
        )

        # 3. Reciprocal Rank Fusion
        fused_docs = self._rrf(
            bm25_docs,
            vector_docs
        )

        # 4. CrossEncoder reranking
        return self._rerank(
            query,
            fused_docs
        )


# =============================================================
# RAG PIPELINE
# =============================================================

class RAGPipeline:

    def __init__(
        self,
        config: Config,
        retriever: HybridRetriever,
        cache: RedisCache,
        vectorStore: VectorStore
    ):

        self.config = config
        self.cache = cache
        self.retriever = retriever
        self.vectorStore = vectorStore

        # -----------------------------------------------------
        # Groq LLM
        # -----------------------------------------------------

        self.llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_token
        )

    # ---------------------------------------------------------
    # Prompt
    # ---------------------------------------------------------

    def _build_prompt(
        self,
        context: str,
        query: str
    ) -> str:

        return f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not in the context, say:
"I could not find the answer in the provided documents."

Give the answer in clear, detailed bullet points.

Context:
{context}

Question:
{query}

Answer:
"""

    # ---------------------------------------------------------
    # Answer question
    # ---------------------------------------------------------

    def answer(
        self,
        query: str
    ) -> str:

        # 1. Check response cache
        cached = self.cache.get_response(
            query
        )

        if cached:

            print(
                "Response served from memory cache"
            )

            return cached

        # 2. Convert question into embedding
        query_vector = self.cache.get_or_embed(
            query,
            self.vectorStore.embdeddin_model
        ).tolist()

        # 3. Hybrid retrieval
        #
        # BM25
        #   +
        # FAISS
        #   ↓
        # RRF
        #   ↓
        # CrossEncoder
        #
        docs = self.retriever.retrieve(
            query,
            query_vector
        )

        # 4. Build context
        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # 5. Generate answer
        response = self.llm.invoke(
            self._build_prompt(
                context,
                query
            )
        ).content

        # 6. Save response in memory cache
        self.cache.set_response(
            query,
            response
        )

        return response