import logging
from google import genai
from app.core.config import settings

logger = logging.getLogger("uvicorn.error")

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768


class Embedding:
    """
    Embedding service using Google Gemini's gemini-embedding-001 model.
    """

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL):
        self.model_name = model_name
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []
        response = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=chunks,
            config={"output_dimensionality": EMBEDDING_DIMENSION}
        )
        return [e.values for e in response.embeddings]

    async def embed_query(self, prompt: str) -> list[float]:
        """Embeds a single query prompt using Gemini embedding model."""
        response = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=prompt,
            config={"output_dimensionality": EMBEDDING_DIMENSION}
        )
        return response.embeddings[0].values


embedding = Embedding()
