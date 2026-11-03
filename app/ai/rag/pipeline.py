
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.ai.rag.data_loader import data_processor
from app.ai.rag.embed import embedding
from app.ai.rag.batch_processor import batch_processor
from app.db.pgvector import pgvector_store
from app.utils.ids import generate_uuid


logger = logging.getLogger("uvicorn.error")


class RagPipeline:

    @staticmethod
    async def ingest_file(path: str | Path):

        path = Path(path)
        uploaded_at = datetime.now(timezone.utc).isoformat()

        # 1. Calculate file hash
        file_hash = data_processor.compute_hash(path)

        # 2. Check whether document already exists
        existing_doc = await pgvector_store.get_document_by_hash(file_hash)

        document_id = None

        if existing_doc:
            document_id = str(existing_doc["document_id"])

            existing_chunks = await pgvector_store.get_chunk_count(
                document_id
            )

            expected_chunks = int(existing_doc.get("chunk_count") or 0)

            # Skip only when all expected chunks are present
            if (
                existing_doc.get("ingestion_status") == "completed"
                and expected_chunks > 0
                and existing_chunks == expected_chunks
            ):
                return {
                    "status": "skipped",
                    "message": (
                        f"File '{path.name}' is already ingested "
                        f"with {existing_chunks} chunks."
                    ),
                    "document": existing_doc
                }

            logger.warning(
                "Document '%s' exists but has %s/%s chunks. Reprocessing.",
                path.name,
                existing_chunks,
                expected_chunks
            )

        # 3. Extract PDF content and generate chunks
        content = await data_processor.process_file(path)

        chunks = content["chunks"]

        if not chunks:
            return {
                "status": "empty",
                "message": f"File '{path.name}' has no extractable text.",
                "document": existing_doc
            }

        # 4. Generate embeddings
        embeddings = await batch_processor.process_embeddings(chunks)

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match "
                f"embedding count ({len(embeddings)})."
            )

        if not embeddings:
            raise ValueError("Embedding generation returned no vectors.")

        # 5. Reuse existing document ID or generate a new one
        if document_id is None:
            document_id = generate_uuid()

        ingested_at = datetime.now(timezone.utc).isoformat()

        document_record = {
            "document_id": document_id,
            "filename": content["filename"],
            "file_hash": file_hash,
            "file_size_bytes": content["file_size_bytes"],
            "uploaded_at": (
                existing_doc.get("uploaded_at")
                if existing_doc
                else uploaded_at
            ),
            "ingested_at": ingested_at,
            "ingestion_status": "completed",
            "chunk_count": len(chunks),
            "chunking_version": "v1",
            "embedding_model": embedding.model_name
        }

        # 6. Store document and chunks transactionally
        await pgvector_store.insert(
            document_record=document_record,
            chunks=chunks,
            embeddings=embeddings
        )

        # 7. Verify insertion
        stored_chunks = await pgvector_store.get_chunk_count(
            document_id
        )

        if stored_chunks != len(chunks):
            raise RuntimeError(
                f"Ingestion verification failed: expected {len(chunks)} "
                f"chunks but found {stored_chunks}."
            )

        logger.info(
            "[RAG] Successfully ingested '%s' with %s chunks.",
            path.name,
            stored_chunks
        )

        return {
            "status": "completed",
            "message": (
                f"File '{path.name}' ingested with "
                f"{stored_chunks} chunks."
            ),
            "document": document_record
        }

    @staticmethod
    async def data_ingestion(folder_path: str | Path = "uploads"):

        files = data_processor.list_files(folder_path)

        summary = {
            "total": len(files),
            "ingested": 0,
            "skipped": 0,
            "empty": 0,
            "failed": 0,
            "results": []
        }

        if not files:
            logger.info("No files to ingest in '%s'", folder_path)
            return summary

        for path in files:

            try:
                result = await RagPipeline.ingest_file(path)

            except Exception as e:
                logger.exception(
                    "Failed to ingest '%s': %s",
                    path.name,
                    e
                )

                result = {
                    "status": "failed",
                    "message": str(e),
                    "document": None
                }

            status = result["status"]

            if status == "completed":
                summary["ingested"] += 1
            elif status == "skipped":
                summary["skipped"] += 1
            elif status == "empty":
                summary["empty"] += 1
            else:
                summary["failed"] += 1

            logger.info("[%s] %s", status, result["message"])

            summary["results"].append({
                "file": path.name,
                **result
            })

        logger.info(
            "Ingestion finished: %s ingested, %s skipped, "
            "%s empty, %s failed (of %s files)",
            summary["ingested"],
            summary["skipped"],
            summary["empty"],
            summary["failed"],
            summary["total"]
        )

        return summary

    @staticmethod
    async def retrieve(
        query: str,
        n_results: int = 5,
        document_id: str | None = None
    ):

        query_embedding = await embedding.embed_query(query)

        results = await pgvector_store.retrieve(
            query_embedding=query_embedding,
            n_results=n_results,
            document_id=document_id
        )

        return {
            "query": query,
            "total_results": len(results),
            "results": results
        }

    @staticmethod
    def build_context(results: list[dict]):

        context_blocks = []
        sources = []
        seen = set()

        for result in results:

            meta = result.get("metadata") or {}

            filename = meta.get("filename", "unknown")
            page_number = meta.get("page_number")

            context_blocks.append(
                f"[Source: {filename}, page {page_number}]\n"
                f"{result['document']}"
            )

            if (filename, page_number) not in seen:

                seen.add((filename, page_number))

                sources.append({
                    "filename": filename,
                    "page_number": page_number,
                    "distance": result.get("distance")
                })

        return "\n\n".join(context_blocks), sources


rag_pipeline = RagPipeline()
