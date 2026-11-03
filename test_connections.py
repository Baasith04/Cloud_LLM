import asyncio
import selectors

from app.db.connections import get_connection


async def main():

    try:
        conn = await get_connection()

        print("PostgreSQL connected successfully!")

        async with conn:

            result = await (
                await conn.execute(
                    "SELECT current_database(), current_user;"
                )
            ).fetchone()

            print("Database:", result[0])
            print("Username:", result[1])

            result = await (
                await conn.execute(
                    """
                    SELECT extname
                    FROM pg_extension
                    WHERE extname = 'vector';
                    """
                )
            ).fetchone()

            print("pgvector extension:", result)

    except Exception as e:
        print("Connection failed:", e)


asyncio.run(
    main(),
    loop_factory=lambda: asyncio.SelectorEventLoop(
        selectors.SelectSelector()
    )
)