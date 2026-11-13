"""Real embedding provider adapter (Gemini Developer API or OpenRouter); no mock fallback."""
from __future__ import annotations

import os
import time
from typing import Protocol

import httpx


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Embedding adapter supporting Gemini (`gemini-embedding-001`) and OpenRouter."""

    def __init__(
        self,
        model_name: str = "openai/text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 50,
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name.strip() or "openai/text-embedding-3-small"
        self.batch_size = batch_size
        self.timeout = timeout
        self.api_requests = 0
        self.embedded_texts = 0
        self.embedded_chars = 0
        self.total_tokens = 0
        self.estimated_cost_usd = 0.0

        if "/" in self.model_name:
            self.provider = "openrouter"
            self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            if not self.api_key:
                raise RuntimeError("OPENROUTER_API_KEY is missing in environment/.env.")
            from openai import OpenAI

            self._openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key, timeout=self.timeout)
            self.dimensions = 1536 if self.model_name.endswith("3-small") else (3072 if self.model_name.endswith("3-large") else dimensions)
        else:
            self.provider = "gemini"
            self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY is missing in environment/.env.")
            self.dimensions = dimensions
            self._http = httpx.Client(timeout=self.timeout)

        # Verify embedding dimensions and connectivity eagerly on init? No, validate first real call
        # or validate dimensions > 0.
        if self.dimensions <= 0:
            raise ValueError(f"Invalid embedding dimensions: {self.dimensions}")

    def _post_gemini_with_retry(self, endpoint: str, payload: dict) -> dict:
        model_path = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:{endpoint}"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        last_error = ""
        for attempt in range(5):
            response = self._http.post(url, headers=headers, json=payload)
            self.api_requests += 1
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2.0 ** attempt, 16.0))
                continue
            raise RuntimeError(f"Gemini embedding request failed ({last_error})")

        raise RuntimeError(f"Gemini embedding request failed after retries ({last_error})")

    def _embed_gemini_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        model_path = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"
        requests_payload = [
            {
                "model": model_path,
                "content": {"parts": [{"text": t}]},
                "taskType": task_type,
                "outputDimensionality": self.dimensions,
            }
            for t in texts
        ]
        data = self._post_gemini_with_retry("batchEmbedContents", {"requests": requests_payload})
        embeddings = [item["values"] for item in data["embeddings"]]
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} embeddings, received {len(embeddings)}")
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if self.provider == "gemini":
                batch_vecs = self._embed_gemini_batch(batch, "RETRIEVAL_DOCUMENT")
            else:
                res = self._openai_client.embeddings.create(model=self.model_name, input=batch)
                self.api_requests += 1
                batch_vecs = [item.embedding for item in res.data]
                if res.usage and getattr(res.usage, "total_tokens", None):
                    tokens = int(res.usage.total_tokens)
                    self.total_tokens += tokens
                    # OpenRouter openai/text-embedding-3-small pricing: $0.02 / 1M tokens
                    if "text-embedding-3-small" in self.model_name:
                        self.estimated_cost_usd += tokens * 0.02 / 1_000_000
            vectors.extend(batch_vecs)
            self.embedded_texts += len(batch)
            self.embedded_chars += sum(len(t) for t in batch)

        for vec in vectors:
            if len(vec) != self.dimensions:
                raise RuntimeError(f"Dimension mismatch: expected {self.dimensions}, got {len(vec)}")
        return vectors

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("Query text cannot be empty.")

        if self.provider == "gemini":
            model_path = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"
            payload = {
                "model": model_path,
                "content": {"parts": [{"text": question}]},
                "taskType": "RETRIEVAL_QUERY",
                "outputDimensionality": self.dimensions,
            }
            data = self._post_gemini_with_retry("embedContent", payload)
            vec = data["embedding"]["values"]
        else:
            res = self._openai_client.embeddings.create(model=self.model_name, input=[question])
            self.api_requests += 1
            vec = res.data[0].embedding
            if res.usage and getattr(res.usage, "total_tokens", None):
                tokens = int(res.usage.total_tokens)
                self.total_tokens += tokens
                if "text-embedding-3-small" in self.model_name:
                    self.estimated_cost_usd += tokens * 0.02 / 1_000_000

        self.embedded_texts += 1
        self.embedded_chars += len(question)
        if len(vec) != self.dimensions:
            raise RuntimeError(f"Query dimension mismatch: expected {self.dimensions}, got {len(vec)}")
        return vec
