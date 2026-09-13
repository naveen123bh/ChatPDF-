
import redis
import hashlib
import numpy as np

from config import Config


class RedisCache:
    def __init__(self, config: Config):
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            username=config.redis_username,
            password=config.redis_password,
            decode_responses=False,
        )

        self.response_cache_ttl = config.response_cache_ttl

        # Check Redis connection immediately
        self.redis_client.ping()
        print("Redis connected successfully")

    # ---------------------------------------------------------
    # Embedding cache
    # ---------------------------------------------------------

    def _embed_key(self, text: str) -> str:
        return f"embedding:{hashlib.md5(text.encode()).hexdigest()}"

    def get_embedding(self, text: str) -> np.ndarray | None:
        raw = self.redis_client.get(
            self._embed_key(text)
        )

        if raw:
            return np.frombuffer(
                raw,
                dtype=np.float32
            )

        return None

    def set_embedding(
        self,
        text: str,
        vector: np.ndarray
    ) -> None:

        self.redis_client.set(
            self._embed_key(text),
            vector.astype(np.float32).tobytes()
        )

    def get_or_embed(
        self,
        text: str,
        encoder
    ):
        cached = self.get_embedding(text)

        if cached is not None:
            print("Embedding served from Redis cache")
            return cached

        vector = encoder.encode(text)

        self.set_embedding(
            text,
            vector
        )

        print("Embedding generated and cached")

        return vector

    # ---------------------------------------------------------
    # Response cache
    # ---------------------------------------------------------

    def _response_key(self, query: str) -> str:
        return (
            f"response:"
            f"{hashlib.md5(query.encode()).hexdigest()}"
        )

    def get_response(
        self,
        query: str
    ) -> str | None:

        raw = self.redis_client.get(
            self._response_key(query)
        )

        print("Checking response cache")

        if raw:
            return raw.decode()

        return None

    def set_response(
        self,
        query: str,
        response: str
    ) -> None:

        self.redis_client.setex(
            self._response_key(query),
            self.response_cache_ttl,
            response,
        )

        print("Response cached in Redis")
