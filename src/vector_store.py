import faiss
import numpy as np

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from src.config import Config


class VectorStore:

    def __init__(
        self,
        config: Config,
        cache=None
    ):
        self.config = config
        self.cache = cache

        # Embedding model
        self.embdeddin_model = SentenceTransformer(
            self.config.embedding_model
        )

        # FAISS cosine-similarity index
        # We normalize vectors and use inner product.
        self.index = faiss.IndexFlatIP(
            self.config.vector_size
        )

        # Keeps the Document corresponding to each FAISS vector
        self.documents: list[Document] = []

    # ---------------------------------------------------------
    # Ingest documents
    # ---------------------------------------------------------

    def ingest(
        self,
        chunks: list[Document]
    ) -> int:

        if not chunks:
            print("No chunks to ingest.")
            return 0

        vectors = []

        for chunk in chunks:

            text = chunk.page_content

            if self.cache is not None:

                vector = self.cache.get_or_embed(
                    text,
                    self.embdeddin_model
                )

            else:

                vector = self.embdeddin_model.encode(
                    text
                )

            vectors.append(vector)

        vectors = np.asarray(
            vectors,
            dtype="float32"
        )

        # Normalize embeddings so inner product = cosine similarity
        faiss.normalize_L2(vectors)

        # Add vectors to FAISS
        self.index.add(vectors)

        # Keep documents in the same order as vectors
        self.documents.extend(chunks)

        print(
            f"Inserted {len(chunks)} chunks into FAISS"
        )

        return len(chunks)

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int
    ) -> list[Document]:

        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(
            [query_vector],
            dtype="float32"
        )

        # Normalize query vector
        faiss.normalize_L2(query_vector)

        # Don't ask FAISS for more vectors than exist
        k = min(
            top_k,
            self.index.ntotal
        )

        scores, indices = self.index.search(
            query_vector,
            k
        )

        results = []

        for index in indices[0]:

            if index < 0:
                continue

            results.append(
                self.documents[index]
            )

        return results