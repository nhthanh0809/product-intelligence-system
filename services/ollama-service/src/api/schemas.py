"""API schemas for Ollama Service."""

from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Model Management Schemas
# =============================================================================


class ModelPullRequest(BaseModel):
    """Request to pull a model from Ollama library."""

    name: str = Field(..., description="Model name to pull (e.g., 'bge-large')")
    insecure: bool = Field(default=False, description="Allow insecure connections")


class ModelPullResponse(BaseModel):
    """Response from model pull operation."""

    name: str
    status: str
    message: str


class ModelInfo(BaseModel):
    """Information about a model."""

    name: str
    size: int | None = None
    digest: str | None = None
    modified_at: str | None = None
    details: dict[str, Any] | None = None


class ModelListResponse(BaseModel):
    """Response containing list of models."""

    models: list[ModelInfo]


class ModelDeleteResponse(BaseModel):
    """Response from model deletion."""

    name: str
    status: str
    message: str


# =============================================================================
# Embedding Schemas
# =============================================================================


class EmbeddingRequest(BaseModel):
    """Request to generate embeddings."""

    texts: list[str] = Field(..., description="List of texts to embed")
    model: str | None = Field(default=None, description="Model to use (defaults to config)")


class EmbeddingResponse(BaseModel):
    """Response containing embeddings."""

    embeddings: list[list[float]]
    model: str
    dimensions: int
    usage: dict[str, int] | None = None


class SingleEmbeddingRequest(BaseModel):
    """Request to generate a single embedding."""

    text: str = Field(..., description="Text to embed")
    model: str | None = Field(default=None, description="Model to use")


class SingleEmbeddingResponse(BaseModel):
    """Response containing a single embedding."""

    embedding: list[float]
    model: str
    dimensions: int


# =============================================================================
# Completion Schemas
# =============================================================================


class GenerateRequest(BaseModel):
    """Request for text generation."""

    prompt: str = Field(..., description="Prompt for generation")
    model: str | None = Field(default=None, description="Model to use")
    system: str | None = Field(default=None, description="System prompt")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    num_predict: int | None = Field(default=None, ge=1)
    stop: list[str] | None = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Stream response")


class GenerateResponse(BaseModel):
    """Response from text generation."""

    response: str
    model: str
    done: bool
    context: list[int] | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class ChatMessage(BaseModel):
    """A chat message."""

    role: str = Field(..., description="Message role: system, user, assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat completion."""

    messages: list[ChatMessage] = Field(..., description="Chat messages")
    model: str | None = Field(default=None, description="Model to use")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    num_predict: int | None = Field(default=None, ge=1)
    stop: list[str] | None = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Stream response")


class ChatResponse(BaseModel):
    """Response from chat completion."""

    message: ChatMessage
    model: str
    done: bool
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


# =============================================================================
# Reranker Schemas
# =============================================================================


class RerankDocument(BaseModel):
    """A document to rerank."""

    text: str = Field(..., description="Document text")
    index: int | None = Field(default=None, description="Original index")


class RerankRequest(BaseModel):
    """Request to rerank documents."""

    query: str = Field(..., description="Query to rank documents against")
    documents: list[str] = Field(..., description="Documents to rerank")
    model: str | None = Field(default=None, description="Reranker model to use")
    top_k: int | None = Field(default=None, description="Return top K results")


class RerankResult(BaseModel):
    """A single rerank result."""

    index: int
    score: float
    text: str


class RerankResponse(BaseModel):
    """Response from reranking."""

    results: list[RerankResult]
    model: str
    usage: dict[str, int] | None = None


# =============================================================================
# Health Check Schemas
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    ollama_status: str
    available_models: list[str]
