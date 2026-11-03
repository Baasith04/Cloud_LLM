import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from app.routes import chat
from contextlib import asynccontextmanager
from app.ai.rag.pipeline import rag_pipeline
from app.db.pgvector import pgvector_store
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("uvicorn.error")

UPLOADS_DIR = Path(__file__).resolve().parents[1] / "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Initializing database tables and pgvector indexes...")
    await pgvector_store.init_db()

    logger.info("App startup: data ingestion is running...")
    await rag_pipeline.data_ingestion(UPLOADS_DIR)

    yield

    logger.info("App shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def landing_page():
    return {"message": "Hi this is your backend"}


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(chat.router)
