"""PostgreSQL client with connection pooling."""

import asyncio
from typing import Any

import asyncpg
import structlog

from src.config import get_settings

logger = structlog.get_logger()


class PostgresClient:
    """PostgreSQL client with asyncpg connection pool."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Initialize connection pool."""
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            host=self.settings.postgres_host,
            port=self.settings.postgres_port,
            database=self.settings.postgres_db,
            user=self.settings.postgres_user,
            password=self.settings.postgres_password,
            min_size=self.settings.postgres_pool_min,
            max_size=self.settings.postgres_pool_max,
            command_timeout=60,
        )
        logger.info("postgres_connected", host=self.settings.postgres_host)

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("postgres_disconnected")

    async def execute(self, query: str, *args) -> str:
        """Execute a query."""
        if not self._pool:
            await self.connect()
        return await self._pool.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[asyncpg.Record]:
        """Fetch multiple rows."""
        if not self._pool:
            await self.connect()
        return await self._pool.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> asyncpg.Record | None:
        """Fetch a single row."""
        if not self._pool:
            await self.connect()
        return await self._pool.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value."""
        if not self._pool:
            await self.connect()
        return await self._pool.fetchval(query, *args)

    async def executemany(self, query: str, args: list[tuple]) -> None:
        """Execute a query with multiple argument sets."""
        if not self._pool:
            await self.connect()
        await self._pool.executemany(query, args)

    async def bulk_upsert_products(
        self,
        products: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> tuple[int, int]:
        """Bulk upsert products into PostgreSQL.

        Returns:
            Tuple of (inserted_count, updated_count)
        """
        if not self._pool:
            await self.connect()

        inserted = 0
        updated = 0

        # Process in batches
        for i in range(0, len(products), batch_size):
            batch = products[i : i + batch_size]

            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    for product in batch:
                        result = await self._upsert_product(conn, product)
                        if result == "INSERT":
                            inserted += 1
                        else:
                            updated += 1

        logger.info(
            "bulk_upsert_completed",
            total=len(products),
            inserted=inserted,
            updated=updated,
        )
        return inserted, updated

    async def _upsert_product(
        self,
        conn: asyncpg.Connection,
        product: dict[str, Any],
    ) -> str:
        """Upsert a single product.

        Returns 'INSERT' or 'UPDATE'.
        """
        # Build column lists and values
        columns = []
        values = []
        placeholders = []
        update_set = []

        for idx, (key, value) in enumerate(product.items(), start=1):
            columns.append(key)
            values.append(value)
            placeholders.append(f"${idx}")
            if key != "asin":  # Don't update primary key
                update_set.append(f"{key} = EXCLUDED.{key}")

        # Add updated_at
        columns.append("updated_at")
        values.append("NOW()")
        placeholders.append("NOW()")
        update_set.append("updated_at = NOW()")

        query = f"""
            INSERT INTO products ({", ".join(columns[:-1])}, updated_at)
            VALUES ({", ".join(placeholders[:-1])}, NOW())
            ON CONFLICT (asin) DO UPDATE SET
                {", ".join(update_set)}
            RETURNING (xmax = 0) as is_insert
        """

        result = await conn.fetchrow(query, *values[:-1])
        return "INSERT" if result["is_insert"] else "UPDATE"

    async def health_check(self) -> dict[str, Any]:
        """Check PostgreSQL health."""
        try:
            if not self._pool:
                await self.connect()

            # Check connection
            version = await self.fetchval("SELECT version()")
            pool_size = self._pool.get_size()
            pool_free = self._pool.get_idle_size()

            return {
                "status": "healthy",
                "version": version,
                "pool_size": pool_size,
                "pool_free": pool_free,
            }
        except Exception as e:
            logger.error("postgres_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Singleton instance
_postgres_client: PostgresClient | None = None


async def get_postgres_client() -> PostgresClient:
    """Get or create PostgreSQL client singleton."""
    global _postgres_client
    if _postgres_client is None:
        _postgres_client = PostgresClient()
        await _postgres_client.connect()
    return _postgres_client
