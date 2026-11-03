import os
import psycopg

from dotenv import load_dotenv
from pgvector.psycopg import register_vector_async

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


async def get_connection():

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is missing in .env")

    conn = await psycopg.AsyncConnection.connect(
        DATABASE_URL
    )

    await register_vector_async(conn)

    return conn