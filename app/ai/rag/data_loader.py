import asyncio
from pathlib import Path
import pymupdf
from app.core.security import get_file_hash

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# ─────────────────────────────────────────────────────────
# Step 1: Load raw text out of files
# ─────────────────────────────────────────────────────────

class DataPreprocessor:


    @staticmethod
    def compute_hash(path: str | Path) -> str:
        """Returns the SHA-256 hash of a file's bytes (used to skip already-ingested files)."""
        return get_file_hash(Path(path).read_bytes())

    @staticmethod
    def load_txt(path: str | Path) -> dict:
        """Reads a plain text / markdown file and returns its contents as a single page."""
        path = Path(path)
        file_bytes = path.read_bytes()
        text = file_bytes.decode("utf-8")

        return {
            "file_path": str(path),
            "filename": path.name,
            "file_hash": get_file_hash(file_bytes),
            "file_size_bytes": len(file_bytes),
            "pages": [{"page_number": 1, "text": text}] if text.strip() else []
        }

    @staticmethod
    def load_pdf(path: str | Path) -> dict:
        """Reads a PDF file and returns its text, one entry per page."""
        path = Path(path)
        file_bytes = path.read_bytes()

        pages = []
        with pymupdf.open(str(path)) as pdf:
            for page in pdf:
                page_text = page.get_text()
                if page_text and page_text.strip():
                    pages.append({
                        "page_number": page.number + 1,
                        "text": page_text
                    })

        return {
            "file_path": str(path),
            "filename": path.name,
            "file_hash": get_file_hash(file_bytes),
            "file_size_bytes": len(file_bytes),
            "pages": pages
        }

    @staticmethod
    def load_file(path: str | Path) -> dict | None:
        """Loads a single file based on its extension. Returns None for unsupported types."""
        extension = Path(path).suffix.lower()

        if extension in (".txt", ".md"):
            return DataPreprocessor.load_txt(path)
        if extension == ".pdf":
            return DataPreprocessor.load_pdf(path)
        return None


    # ─────────────────────────────────────────────────────────
    # Step 2: Split page text into small overlapping chunks
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def chunk_text(pages: list[dict], chunk_size: int = 300, overlap: int = 50) -> list[dict]:
        """
        Splits each page into chunks of `chunk_size` words, with `overlap` words
        repeated between consecutive chunks so we don't cut a sentence's
        meaning in half at a chunk boundary.
        """
        chunks = []
        chunk_index = 0
        step = chunk_size - overlap

        for page in pages:
            page_number = page["page_number"]
            words = page["text"].split()
            if not words:
                continue  # skip empty pages, keep processing the rest

            for start in range(0, len(words), step):
                chunk_words = words[start:start + chunk_size]
                chunks.append({
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "text": " ".join(chunk_words)
                })
                chunk_index += 1

                if start + chunk_size >= len(words):
                    break  # reached the end of the page

        return chunks


    # ─────────────────────────────────────────────────────────
    # Step 3: Put it together — file -> chunks ready to embed
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def list_files(folder_path: str | Path) -> list[Path]:
        """Returns every supported file in a folder (creates the folder if missing)."""
        folder = Path(folder_path)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            return []

        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    @staticmethod
    def _process_file(path: str | Path) -> dict:
        content = DataPreprocessor.load_file(path)
        if content is None:
            raise ValueError(f"Unsupported file type: {path}")

        chunks = DataPreprocessor.chunk_text(content["pages"])
        content["chunks"] = chunks
        content["total_chunks"] = len(chunks)
        return content

    @staticmethod
    async def process_file(path: str | Path) -> dict:
        """
        Loads a file and splits it into chunks (runs in a thread so parsing
        doesn't block the event loop).

        Returns a dict like:
            {"filename": ..., "file_hash": ..., "file_size_bytes": ...,
             "pages": [...], "chunks": [{"chunk_index", "page_number", "text"}], "total_chunks": N}
        """
        return await asyncio.to_thread(DataPreprocessor._process_file, path)


data_processor = DataPreprocessor()
