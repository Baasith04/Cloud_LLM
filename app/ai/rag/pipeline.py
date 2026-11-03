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

        # 1. Existing check — skip before parsing if this content is already stored
        file_hash = data_processor.compute_hash(path)
        existing_doc = await pgvector_store.get_document_by_hash(file_hash)
        if existing_doc:
            return {
                "status": "skipped",
                "message": f"File '{path.name}' already ingested as '{existing_doc.get('filename')}'.",
                "document": existing_doc
            }

        # 2. Process file, extract text per page and chunk
        content = await data_processor.process_file(path)
        chunks = content["chunks"]
        if not chunks:
            return {
                "status": "empty",
                "message": f"File '{path.name}' has no extractable text.",
                "document": None
            }

        # 3. Generate embeddings for the chunks using batch processing
        embeddings = await batch_processor.process_embeddings(chunks)

        # 4. Construct Document Record
        document_id = generate_uuid()
        ingested_at = datetime.now(timezone.utc).isoformat()
        document_record = {
            "document_id": document_id,
            "filename": content["filename"],
            "file_hash": file_hash,
            "file_size_bytes": content["file_size_bytes"],
            "uploaded_at": uploaded_at,
            "ingested_at": ingested_at,
            "ingestion_status": "completed",
            "chunk_count": content["total_chunks"],
            "chunking_version": "v1",
            "embedding_model": embedding.model_name
        }

        # 5. Insert into PostgreSQL + pgvector
        await pgvector_store.insert(
            document_record=document_record,
            chunks=chunks,
            embeddings=embeddings
        )

        return {
            "status": "completed",
            "message": f"File '{path.name}' ingested with {content['total_chunks']} chunks.",
            "document": document_record
        }

    @staticmethod
    async def data_ingestion(folder_path: str | Path = "uploads"):
        """Ingests every supported file in a folder, skipping files whose hash is already in the DB."""

        files = data_processor.list_files(folder_path)
        summary = {

            "total": len(files), 
            "ingested": 0, 
            "skipped": 0, 
            "failed": 0, 
            "results": []
        }

        if not files:
            logger.info(f"No files to ingest in '{folder_path}'")
            return summary

        for path in files:
            try:
                result = await RagPipeline.ingest_file(path)
            except Exception as e:
                logger.exception(f"Failed to ingest '{path.name}': {e}")
                result = {"status": "failed", "message": str(e), "document": None}

            status = result["status"]
            if status == "completed":
                summary["ingested"] += 1
            elif status == "failed":
                summary["failed"] += 1
            else:
                summary["skipped"] += 1

            logger.info(f"[{status}] {result['message']}")
            summary["results"].append({"file": path.name, **result})

        logger.info(
            f"Ingestion finished: {summary['ingested']} ingested, "
            f"{summary['skipped']} skipped, {summary['failed']} failed "
            f"(of {summary['total']} files)"
        )
        return summary

    @staticmethod
    async def retrieve(query: str, n_results: int = 5, document_id: str | None = None):
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
    def build_context(results: list[dict]) -> tuple[str, list[dict]]:
        """Formats retrieved chunks into a context string for the LLM, plus a list of unique sources."""

        context_blocks = []
        sources = []
        seen = set()

        for result in results:
            meta = result.get("metadata") or {}
            filename = meta.get("filename", "unknown")
            page_number = meta.get("page_number")

            context_blocks.append(f"[Source: {filename}, page {page_number}]\n{result['document']}")

            if (filename, page_number) not in seen:
                seen.add((filename, page_number))
                sources.append({
                    "filename": filename,
                    "page_number": page_number,
                    "distance": result.get("distance")
                })

        return "\n\n".join(context_blocks), sources


rag_pipeline = RagPipeline()
