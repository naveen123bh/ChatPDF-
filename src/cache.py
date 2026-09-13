import hashlib
import numpy as np

from src.config import Config


class RedisCache:

    def __init__(self, config: Config):
        self.config = config

        # In-memory embedding cache
        self.embedding_cache = {}

        # In-memory response cache
        self.response_cache = {}

        print("In-memory cache initialized")

    # ---------------------------------------------------------
    # Embedding cache
    # ---------------------------------------------------------

    def _embed_key(self, text: str) -> str:
        return hashlib.md5(
            text.encode()
        ).hexdigest()

    def get_embedding(
        self,
        text: str
    ) -> np.ndarray | None:

        key = self._embed_key(text)

        return self.embedding_cache.get(key)

    def set_embedding(
        self,
        text: str,
        vector: np.ndarray
    ) -> None:

        key = self._embed_key(text)

        self.embedding_cache[key] = np.asarray(
            vector,
            dtype=np.float32
        )

    def get_or_embed(
        self,
        text: str,
        encoder
    ):

        cached = self.get_embedding(text)

        if cached is not None:

            print(
                "Embedding served from memory cache"
            )

            return cached

        vector = encoder.encode(text)

        vector = np.asarray(
            vector,
            dtype=np.float32
        )

        self.set_embedding(
            text,
            vector
        )

        print(
            "Embedding generated and cached"
        )

        return vector

    # ---------------------------------------------------------
    # Response cache
    # ---------------------------------------------------------

    def _response_key(
        self,
        query: str
    ) -> str:

        return hashlib.md5(
            query.encode()
        ).hexdigest()

    def get_response(
        self,
        query: str
    ) -> str | None:

        key = self._response_key(query)

        print(
            "Checking response cache"
        )

        return self.response_cache.get(key)

    def set_response(
        self,
        query: str,
        response: str
    ) -> None:

        key = self._response_key(query)

        self.response_cache[key] = response

        print(
            "Response cached in memory"
        )