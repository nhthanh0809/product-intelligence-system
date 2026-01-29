"""Data Pipeline Service - FastAPI Application."""

import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.api.routes import router
from src.clients.postgres_client import get_postgres_client
from src.clients.qdrant_client import get_qdrant_client
from src.clients.elasticsearch_client import get_elasticsearch_client
from src.clients.ollama_client import get_ollama_client
from src.clients.redis_client import get_redis_client

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info(
        "service_starting",
        service=settings.service_name,
        port=settings.service_port,
    )

    # Initialize clients
    try:
        # PostgreSQL
        postgres = await get_postgres_client()
        logger.info("postgres_initialized")

        # Qdrant
        qdrant = get_qdrant_client()
        await qdrant.ensure_collection()
        logger.info("qdrant_initialized")

        # Elasticsearch
        es = await get_elasticsearch_client()
        await es.ensure_index()
        logger.info("elasticsearch_initialized")

        # Ollama
        ollama = await get_ollama_client()
        logger.info("ollama_initialized")

        # Redis
        redis = await get_redis_client()
        logger.info("redis_initialized")

    except Exception as e:
        logger.error("initialization_failed", error=str(e))
        # Continue anyway - health check will show degraded status

    logger.info("service_started")

    yield

    # Cleanup
    logger.info("service_stopping")

    try:
        # Close all connections
        postgres = await get_postgres_client()
        await postgres.close()

        qdrant = get_qdrant_client()
        qdrant.close()

        es = await get_elasticsearch_client()
        await es.close()

        ollama = await get_ollama_client()
        await ollama.close()

        redis = await get_redis_client()
        await redis.close()

    except Exception as e:
        logger.error("cleanup_failed", error=str(e))

    logger.info("service_stopped")


# Create FastAPI app
app = FastAPI(
    title="Data Pipeline Service",
    description="Simplified pipeline: CSV -> Extract -> Clean -> Embed -> Load (Postgres, Qdrant, Elasticsearch)",
    version="0.2.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.service_name,
        "version": "0.1.0",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=True,
    )
