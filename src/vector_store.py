from qdrant_client import QdrantClient
from langchain_core.documents import Document
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from cache import RedisCache
import hashlib
from config import Config

class VectorStore:
    def __init__(self,config:Config,cache:RedisCache):
        self.config = config
        self.client = QdrantClient(
            url = config.qdrant_url,
            api_key = config.qdrant_api_key
        )
        self.cache = cache
        self.embdeddin_model = SentenceTransformer(self.config.embedding_model)
        self._ensure_collection()
    def _ensure_collection(self)->None:
       collections = self.client.get_collections().collections
       collections_names = [c.name for c in collections]
       if self.config.collection_name not in collections_names:
           self.client.create_collection(
               collection_name = self.config.collection_name,
               vectors_config = VectorParams(
                   size = self.config.vector_size,
                   distance = self.config.vector_distance
               )
           )
    @staticmethod
    def _make_id(source:str, page:int, text:str)->str:
        return hashlib.md5(f"{source}_{page}_{text}".encode()).hexdigest()
    
    def _exists(self, point_id: str) -> bool:
        return len(self.client.retrieve(
            collection_name=self.config.collection_name,
            ids=[point_id],
            with_vectors=False,
            with_payload=False,
        )) > 0
    
    def ingest(self, chunks: list[Document]) -> int:
        points = []
        for chunk in chunks:
            text = chunk.page_content
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            pid = self._make_id(source, page, text)

            if self._exists(pid):
                continue

            vector = self.cache.get_or_embed(text, self.embdeddin_model)

            points.append(PointStruct(
                id=pid,
                vector=vector.tolist(),
                payload={"text": text, "source": source, "page": page},
            ))

        # ── Batch upsert instead of one giant request ──
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
            )
            print(f"Upserted batch {i // batch_size + 1} — {len(batch)} points")

        print(f"Inserted {len(points)} new chunks into Qdrant")
        return len(points)
    
    def search(self, query_vector: list[float], top_k: int) -> list[Document]:
        results = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            limit=top_k,
        )
        return [
            Document(
                page_content=p.payload["text"],
                metadata={"source": p.payload["source"], "page": p.payload["page"]},
            )
            for p in results.points
        ]