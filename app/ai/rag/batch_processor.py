import asyncio
import logging
from google import genai
from app.core.config import settings

logger = logging.getLogger("uvicorn.error")

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_CONCURRENCY = 2
EMBEDDING_DIMENSION = 768


class BatchProcessor:
    """
    Simple and robust batch processing with concurrency for Gemini embeddings.
    """

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL):
        self.model_name = model_name
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def _embed_single_batch(
        self,
        batch_chunks: list[dict],
        batch_num: int,
        total_batches: int,
        semaphore: asyncio.Semaphore
    ) -> list[list[float]]:
        """Processes a single batch under the concurrency semaphore."""
        async with semaphore:
            batch_texts = [c["text"] for c in batch_chunks]
            logger.info(f"[Batch {batch_num}/{total_batches}] Embedding {len(batch_texts)} chunks...")

            response = await self.client.aio.models.embed_content(
                model=self.model_name,
                contents=batch_texts,
                config={"output_dimensionality": EMBEDDING_DIMENSION}
            )
            return [e.values for e in response.embeddings]

    async def process_embeddings(
        self,
        chunks: list[dict],
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    ) -> list[list[float]]:
        """
        1. Batch Processing: Slices chunks into batches of size `batch_size`.
        2. Concurrency: Executes batch requests concurrently using asyncio.gather + Semaphore.
        """
        if not chunks:
            return []

        # 1. Batch Processing: Divide chunks into sub-lists
        batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
        total_batches = len(batches)

        logger.info(
            f"Processing {len(chunks)} chunks in {total_batches} batches "
            f"(batch_size={batch_size}, max_concurrency={max_concurrency}) via {self.model_name}"
        )

        # 2. Concurrency: Controlled parallel execution
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [
            self._embed_single_batch(batch, idx + 1, total_batches, semaphore)
            for idx, batch in enumerate(batches)
        ]

        # Gather results concurrently while preserving exact batch order
        batch_results = await asyncio.gather(*tasks)

        # Flatten into a single list of embeddings
        all_embeddings = [emb for batch in batch_results for emb in batch]
        logger.info(f"Finished embedding all {len(all_embeddings)} chunks successfully.")
        return all_embeddings


batch_processor = BatchProcessor()
