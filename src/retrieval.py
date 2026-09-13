
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from vector_store import VectorStore
from config import Config
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from cache import RedisCache


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

        # BM25 lexical retrieval
        self.bm25 = BM25Retriever.from_documents(
            chunks,
            k=self.config.vector_top_k
        )

        # CrossEncoder reranker
        self.reranker = CrossEncoder(config.reranker_model)

    # Reciprocal Rank Fusion
    def _rrf(
        self,
        bm25_docs: list[Document],
        vector_docs: list[Document]
    ):
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        k = self.config.rrf_k

        for rank, doc in enumerate(bm25_docs):
            c = doc.page_content
            doc_map[c] = doc
            scores[c] = scores.get(c, 0) + 1 / (k + rank + 1)

        for rank, doc in enumerate(vector_docs):
            c = doc.page_content
            doc_map[c] = doc
            scores[c] = scores.get(c, 0) + 1 / (k + rank + 1)

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc_map[c] for c, _ in ranked]

    def _rerank(
        self,
        query: str,
        docs: list[Document]
    ) -> list[Document]:

        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        scores = self.reranker.predict(pairs)

        scored = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        top = scored[:self.config.rerank_top_k]

        for i, (doc, score) in enumerate(top, 1):
            print(
                f"Rank {i} | "
                f"Score: {score:.4f} | "
                f"{doc.page_content[:80]}..."
            )

        return [doc for doc, _ in top]

    def retrieve(
        self,
        query: str,
        query_vector: list[float]
    ) -> list[Document]:

        # 1. Lexical retrieval
        bm25_docs = self.bm25.invoke(query)

        # 2. Semantic/vector retrieval
        vector_docs = self.vectorStore.search(
            query_vector,
            self.config.vector_top_k
        )

        # 3. Reciprocal Rank Fusion
        fused = self._rrf(
            bm25_docs,
            vector_docs
        )

        # 4. CrossEncoder reranking
        return self._rerank(
            query,
            fused
        )


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

        self.llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_token
        )

    def _build_prompt(
        self,
        context: str,
        query: str
    ) -> str:

        return f"""You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not in the context, say:
"I could not find the answer in the provided documents."

Give answers in detailed bullet points.

Context:
{context}

Question:
{query}

Answer:"""

    def answer(self, query: str) -> str:

        # 1. Check response cache first
        cached = self.cache.get_response(query)

        if cached:
            print("Response served from Redis cache")
            return cached

        # 2. Embed query
        # Redis embedding cache is used internally
        query_vector = self.cache.get_or_embed(
            query,
            self.vectorStore.embdeddin_model
        ).tolist()

        # 3. Hybrid retrieval
        # BM25 + Vector → RRF → CrossEncoder
        docs = self.retriever.retrieve(
            query,
            query_vector
        )

        # 4. Build context
        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # 5. Generate answer with Groq
        response = self.llm.invoke(
            self._build_prompt(
                context,
                query
            )
        ).content

        # 6. Cache response
        self.cache.set_response(
            query,
            response
        )

        return response