"""API routes for Ollama Service."""

from fastapi import APIRouter, HTTPException
import structlog

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelDeleteResponse,
    ModelInfo,
    ModelListResponse,
    ModelPullRequest,
    ModelPullResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    SingleEmbeddingRequest,
    SingleEmbeddingResponse,
)
from src.config import get_settings
from src.services.completion import get_completion_service
from src.services.embedding import get_embedding_service
from src.services.model_manager import get_model_manager

logger = structlog.get_logger()
router = APIRouter()


# =============================================================================
# Health Check
# =============================================================================


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    settings = get_settings()
    model_manager = get_model_manager()

    health = await model_manager.health_check()

    return HealthResponse(
        status="healthy" if health["status"] == "healthy" else "unhealthy",
        service=settings.service_name,
        ollama_status=health["status"],
        available_models=health.get("available_models", []),
    )


# =============================================================================
# Model Management
# =============================================================================


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all available models."""
    model_manager = get_model_manager()
    models = await model_manager.list_models()

    return ModelListResponse(
        models=[
            ModelInfo(
                name=m.get("name", ""),
                size=m.get("size"),
                digest=m.get("digest"),
                modified_at=m.get("modified_at"),
                details=m.get("details"),
            )
            for m in models
        ]
    )


@router.get("/models/{name}/info", response_model=ModelInfo)
async def get_model_info(name: str):
    """Get information about a specific model."""
    model_manager = get_model_manager()
    info = await model_manager.get_model_info(name)

    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

    return ModelInfo(
        name=name,
        details=info,
    )


@router.post("/models/pull", response_model=ModelPullResponse)
async def pull_model(request: ModelPullRequest):
    """Pull a model from Ollama library."""
    model_manager = get_model_manager()

    try:
        result = await model_manager.pull_model(request.name, request.insecure)

        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to pull model: {result.get('error')}",
            )

        return ModelPullResponse(
            name=request.name,
            status="success",
            message=f"Model '{request.name}' pulled successfully",
        )
    except Exception as e:
        logger.error("pull_model_failed", model=request.name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{name}", response_model=ModelDeleteResponse)
async def delete_model(name: str):
    """Delete a model."""
    model_manager = get_model_manager()

    try:
        result = await model_manager.delete_model(name)

        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

        return ModelDeleteResponse(
            name=name,
            status="success",
            message=f"Model '{name}' deleted successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_model_failed", model=name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Embeddings
# =============================================================================


@router.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for multiple texts."""
    embedding_service = get_embedding_service()

    try:
        embeddings = await embedding_service.embed_batch(
            request.texts,
            model=request.model,
        )

        model = request.model or get_settings().default_embedding_model
        dimensions = embedding_service.get_dimensions(model)

        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            dimensions=dimensions,
            usage={"total_texts": len(request.texts)},
        )
    except Exception as e:
        logger.error("embedding_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed/single", response_model=SingleEmbeddingResponse)
async def generate_single_embedding(request: SingleEmbeddingRequest):
    """Generate embedding for a single text."""
    embedding_service = get_embedding_service()

    try:
        embedding = await embedding_service.embed(
            request.text,
            model=request.model,
        )

        model = request.model or get_settings().default_embedding_model
        dimensions = embedding_service.get_dimensions(model)

        return SingleEmbeddingResponse(
            embedding=embedding,
            model=model,
            dimensions=dimensions,
        )
    except Exception as e:
        logger.error("embedding_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Completions
# =============================================================================


@router.post("/generate", response_model=GenerateResponse)
async def generate_completion(request: GenerateRequest):
    """Generate text completion."""
    completion_service = get_completion_service()

    try:
        result = await completion_service.generate(
            prompt=request.prompt,
            model=request.model,
            system=request.system,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            num_predict=request.num_predict,
            stop=request.stop,
            stream=False,  # Non-streaming for this endpoint
        )

        return GenerateResponse(
            response=result.get("response", ""),
            model=result.get("model", request.model or get_settings().default_llm_model),
            done=result.get("done", True),
            context=result.get("context"),
            total_duration=result.get("total_duration"),
            load_duration=result.get("load_duration"),
            prompt_eval_count=result.get("prompt_eval_count"),
            eval_count=result.get("eval_count"),
            eval_duration=result.get("eval_duration"),
        )
    except Exception as e:
        logger.error("generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Generate chat completion."""
    completion_service = get_completion_service()

    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        result = await completion_service.chat(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            num_predict=request.num_predict,
            stop=request.stop,
            stream=False,  # Non-streaming for this endpoint
        )

        message = result.get("message", {})

        return ChatResponse(
            message=ChatMessage(
                role=message.get("role", "assistant"),
                content=message.get("content", ""),
            ),
            model=result.get("model", request.model or get_settings().default_llm_model),
            done=result.get("done", True),
            total_duration=result.get("total_duration"),
            load_duration=result.get("load_duration"),
            prompt_eval_count=result.get("prompt_eval_count"),
            eval_count=result.get("eval_count"),
            eval_duration=result.get("eval_duration"),
        )
    except Exception as e:
        logger.error("chat_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Reranking
# =============================================================================


@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Rerank documents using BGE reranker model.

    Uses cross-encoder approach: scores each query-document pair.
    """
    import asyncio
    import httpx

    settings = get_settings()
    model = request.model or "qllama/bge-reranker-v2-m3:latest"

    async def score_document(idx: int, doc: str) -> RerankResult:
        """Score a single document against the query."""
        # Truncate long documents
        doc_truncated = doc[:1500] if doc else ""

        # BGE reranker format
        rerank_input = f"query: {request.query} document: {doc_truncated}"

        try:
            async with httpx.AsyncClient() as client:
                # Use Ollama embeddings API
                response = await client.post(
                    f"{settings.ollama_host}/api/embeddings",
                    json={
                        "model": model,
                        "prompt": rerank_input,
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])

                    # BGE reranker v2-m3: relevance score is at position 297
                    # This was empirically determined by testing the model
                    if embedding and len(embedding) > 297:
                        score = embedding[297]
                        # Normalize score to 0-1 range (typical scores range from 0-10)
                        score = max(0.0, min(1.0, score / 10.0))
                    else:
                        score = 0.0
                else:
                    logger.warning("rerank_api_error", status=response.status_code, doc_idx=idx)
                    score = 0.0

        except Exception as e:
            logger.debug("rerank_item_failed", index=idx, error=str(e))
            score = 0.0

        return RerankResult(index=idx, score=float(score), text=doc)

    try:
        # Process documents in parallel batches
        batch_size = 5
        results = []

        for batch_start in range(0, len(request.documents), batch_size):
            batch_end = min(batch_start + batch_size, len(request.documents))
            batch_docs = request.documents[batch_start:batch_end]

            tasks = [
                score_document(batch_start + i, doc)
                for i, doc in enumerate(batch_docs)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k if specified
        if request.top_k is not None:
            results = results[:request.top_k]

        return RerankResponse(
            results=results,
            model=model,
            usage={"total_documents": len(request.documents)},
        )

    except Exception as e:
        logger.error("rerank_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
