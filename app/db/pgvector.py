
import logging

from psycopg.rows import dict_row

from app.db.connections import get_connection
from app.utils.ids import generate_uuid


logger = logging.getLogger("uvicorn.error")


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL,
    ingestion_status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    chunking_version TEXT NOT NULL,
    embedding_model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id UUID NOT NULL
        REFERENCES documents(document_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx
ON chunks (document_id);

CREATE INDEX IF NOT EXISTS documents_file_hash_idx
ON documents (file_hash);
"""


class PgVectorStore:
    """PostgreSQL + pgvector storage implementation for RAG."""

    @staticmethod
    async def init_db() -> None:
        """Initialize the pgvector extension, tables, and indexes."""

        async with await get_connection() as conn:
            await conn.execute(SCHEMA_SQL)

        logger.info(
            "[pgvector] Database tables and HNSW index initialized."
        )

    @staticmethod
    async def _fetch_document(
        column: str,
        value: str
    ) -> dict | None:
        """Fetch a document and format datetime values."""

        allowed_columns = {"document_id", "file_hash"}

        if column not in allowed_columns:
            raise ValueError(f"Invalid document column: {column}")

        async with await get_connection() as conn:
            conn.row_factory = dict_row

            cur = await conn.execute(
                f"SELECT * FROM documents WHERE {column} = %s LIMIT 1;",
                (value,)
            )

            row = await cur.fetchone()

            if row:
                for field in ("uploaded_at", "ingested_at"):
                    if hasattr(row[field], "isoformat"):
                        row[field] = row[field].isoformat()
                    else:
                        row[field] = str(row[field])

            return row

    @staticmethod
    async def get_document_by_hash(
        file_hash: str
    ) -> dict | None:
        return await PgVectorStore._fetch_document(
            "file_hash",
            file_hash
        )

    @staticmethod
    async def get_document_by_id(
        document_id: str
    ) -> dict | None:
        return await PgVectorStore._fetch_document(
            "document_id",
            document_id
        )

    @staticmethod
    async def get_chunk_count(document_id: str) -> int:
        """Return the actual number of chunks stored for a document."""

        async with await get_connection() as conn:
            conn.row_factory = dict_row

            cur = await conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM chunks
                WHERE document_id = %s;
                """,
                (document_id,)
            )

            row = await cur.fetchone()

            return row["total"] if row else 0

    @staticmethod
    async def insert(
        document_record: dict,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> None:
        """
        Atomically insert document metadata and chunk embeddings.

        Existing chunks for the document are replaced during re-ingestion.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match "
                f"embeddings count ({len(embeddings)})"
            )

        if not chunks:
            raise ValueError("Cannot insert a document without chunks.")

        if not embeddings:
            raise ValueError("Cannot insert a document without embeddings.")

        async with await get_connection() as conn:
            async with conn.transaction():

                # 1. Insert or update document metadata
                await conn.execute(
                    """
                    INSERT INTO documents (
                        document_id,
                        filename,
                        file_hash,
                        file_size_bytes,
                        uploaded_at,
                        ingested_at,
                        ingestion_status,
                        chunk_count,
                        chunking_version,
                        embedding_model
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)

                    ON CONFLICT (document_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        ingested_at = EXCLUDED.ingested_at,
                        ingestion_status = EXCLUDED.ingestion_status,
                        chunk_count = EXCLUDED.chunk_count,
                        chunking_version = EXCLUDED.chunking_version,
                        embedding_model = EXCLUDED.embedding_model;
                    """,
                    (
                        document_record["document_id"],
                        document_record["filename"],
                        document_record["file_hash"],
                        int(document_record["file_size_bytes"]),
                        document_record["uploaded_at"],
                        document_record["ingested_at"],
                        document_record["ingestion_status"],
                        int(document_record["chunk_count"]),
                        document_record["chunking_version"],
                        document_record["embedding_model"],
                    )
                )

                # 2. Delete existing chunks for re-ingestion
                await conn.execute(
                    """
                    DELETE FROM chunks
                    WHERE document_id = %s;
                    """,
                    (document_record["document_id"],)
                )

                # 3. Prepare new chunk rows
                chunk_rows = [
                    (
                        str(
                            chunk.get("id")
                            or chunk.get("chunk_id")
                            or generate_uuid()
                        ),
                        document_record["document_id"],
                        document_record["filename"],
                        int(chunk["chunk_index"]),
                        int(chunk["page_number"]),
                        chunk["text"],
                        embedding
                    )
                    for chunk, embedding in zip(chunks, embeddings)
                ]

                # 4. Insert new chunks and embeddings
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """
                        INSERT INTO chunks (
                            chunk_id,
                            document_id,
                            filename,
                            chunk_index,
                            page_number,
                            content,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """,
                        chunk_rows
                    )

        logger.info(
            "[pgvector] Ingested '%s' (%d chunks).",
            document_record["filename"],
            len(chunks)
        )

    @staticmethod
    async def retrieve(
        query_embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None
    ) -> list[dict]:
        """Retrieve the most relevant chunks using cosine similarity."""

        if document_id:
            where_filter = "WHERE document_id = %s"
            params = (
                query_embedding,
                document_id,
                n_results
            )
        else:
            where_filter = ""
            params = (
                query_embedding,
                n_results
            )

        query = f"""
            SELECT
                chunk_id,
                document_id,
                filename,
                chunk_index,
                page_number,
                content,
                embedding <=> %s::vector AS distance
            FROM chunks
            {where_filter}
            ORDER BY distance
            LIMIT %s;
        """

        async with await get_connection() as conn:
            conn.row_factory = dict_row

            cur = await conn.execute(query, params)
            rows = await cur.fetchall()

            return [
                {
                    "id": row["chunk_id"],
                    "document": row["content"],
                    "metadata": {
                        "document_id": row["document_id"],
                        "filename": row["filename"],
                        "chunk_index": row["chunk_index"],
                        "page_number": row["page_number"],
                    },
                    "distance": float(row["distance"])
                }
                for row in rows
            ]


pgvector_store = PgVectorStore()
