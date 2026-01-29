"""Main FastAPI application for Ollama Service."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import get_settings
from src.services.completion import get_completion_service
from src.services.embedding import get_embedding_service
from src.services.model_manager import get_model_manager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    logger.info("starting_service", service=settings.service_name, port=settings.service_port)

    # Initialize services
    model_manager = get_model_manager()
    embedding_service = get_embedding_service()
    completion_service = get_completion_service()

    # Check Ollama health
    health = await model_manager.health_check()
    logger.info("ollama_status", status=health["status"], models=health.get("available_models", []))

    yield

    # Cleanup
    logger.info("shutting_down_service", service=settings.service_name)
    await model_manager.close()
    await embedding_service.close()
    await completion_service.close()


# Create FastAPI application
settings = get_settings()

app = FastAPI(
    title="Ollama Model Service",
    description="Model hosting and management service for Product Intelligence System",
    version="0.1.0",
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

# Include routes
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
