from langchain_core.documents import Document
from langchain_groq import ChatGroq
from vector_store import VectorStore
from config import Config
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from cache import RedisCache

class HybridRetriever:
    def __init__(self,chunks:list[Document], vectorStore: VectorStore, config:Config):
        self.chunks = chunks
        self.vectorStore = vectorStore
        self.config = config
        self.bm25 = BM25Retriever.from_documents(chunks)
        self.reranker = CrossEncoder(config.reranker_model)

    #Recipocal Rak Fusion
    def _rrf(self, bm25_docs:list[Document], vector_docs:list[Document]):
        scores:dict[str,float] = {}
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

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[c] for c, _ in ranked]
    

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top = scored[: self.config.rerank_top_k]

        for i, (doc, score) in enumerate(top, 1):
            print(f"Rank {i} | Score: {score:.4f} | {doc.page_content[:80]}...")

        return [doc for doc, _ in top]
    
    def retrieve(self, query: str, query_vector: list[float]) -> list[Document]:
        bm25_docs = self.bm25.invoke(query)
        vector_docs = self.vectorStore.search(query_vector, self.config.vector_top_k)
        fused = self._rrf(bm25_docs, vector_docs)
        return self._rerank(query, fused)

class RAGPipeline:
    def __init__(self,config:Config, retriever:HybridRetriever, cache:RedisCache, vectorStore:VectorStore):
        self.config = config
        self.cache = cache
        self.retriever = retriever
        self.cache = cache
        self.vectorStore = vectorStore
        self.llm = ChatGroq(
            api_key= self.config.groq_api_key,
            model = self.config.llm_model,
            temperature = self.config.llm_temperature,
            max_tokens = self.config.max_token
        )

    def _build_prompt(self, context: str, query: str) -> str:
        return f"""You are a helpful AI assistant.
        Answer ONLY from the provided context.
        If the answer is not in the context, say: "I could not find the answer in the provided documents."
        Give answers in detailed bullet points.
        Context:
        {context}
        Question:
        {query}
        Answer:"""
    
    def answer(self, query:str)->str:
        cached = self.cache.get_response(query)
        #1.Check response cache first
        if cached:
            print("Response served from Redis cache")
            return cached
        
        # 2. Embed query (Redis embedding cache used internally)
        query_vector =self.cache.get_or_embed(query,self.vectorStore.embdeddin_model).tolist()

        # 3. Hybrid retrieve → RRF → rerank
        docs = self.retriever.retrieve(query, query_vector)
        context = "\n\n".join(doc.page_content for doc in docs)

        # 4. LLM
        response = self.llm.invoke(self._build_prompt(context, query)).content

        # 5. Cache the response
        self.cache.set_response(query, response)

        return response