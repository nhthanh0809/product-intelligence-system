"""Completion service for Ollama."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.services.model_manager import get_model_manager

logger = structlog.get_logger()


class CompletionService:
    """Service for text and chat completions via Ollama."""

    def __init__(self):
        """Initialize completion service."""
        self.settings = get_settings()
        self.base_url = self.settings.ollama_host
        self._client: httpx.AsyncClient | None = None
        self._model_manager = get_model_manager()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=self.settings.connect_timeout,
                    read=self.settings.read_timeout,
                    write=30.0,
                    pool=30.0,
                ),
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_options(
        self,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_predict: int | None = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build options dict for Ollama API."""
        options = {}

        if temperature is not None:
            options["temperature"] = temperature
        else:
            options["temperature"] = self.settings.default_temperature

        if top_p is not None:
            options["top_p"] = top_p
        else:
            options["top_p"] = self.settings.default_top_p

        if top_k is not None:
            options["top_k"] = top_k
        else:
            options["top_k"] = self.settings.default_top_k

        if num_predict is not None:
            options["num_predict"] = num_predict
        else:
            options["num_predict"] = self.settings.default_num_predict

        if stop:
            options["stop"] = stop

        return options

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_predict: int | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Generate text completion.

        Args:
            prompt: Prompt text
            model: Model to use
            system: System prompt
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            num_predict: Max tokens to generate
            stop: Stop sequences
            stream: Whether to stream response

        Returns:
            Generation response
        """
        model = model or self.settings.default_llm_model

        # Ensure model is available
        await self._model_manager.ensure_model(model)

        client = await self._get_client()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": self._build_options(temperature, top_p, top_k, num_predict, stop),
        }

        if system:
            payload["system"] = system

        logger.debug("generating", model=model, prompt_length=len(prompt))

        response = await client.post("/api/generate", json=payload)
        response.raise_for_status()

        if stream:
            # For streaming, return the response object
            return {"streaming": True, "response": response}

        data = response.json()
        logger.debug(
            "generation_complete",
            model=model,
            response_length=len(data.get("response", "")),
            eval_count=data.get("eval_count"),
        )

        return data

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_predict: int | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream text generation.

        Args:
            prompt: Prompt text
            model: Model to use
            system: System prompt
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            num_predict: Max tokens to generate
            stop: Stop sequences

        Yields:
            Generated text chunks
        """
        model = model or self.settings.default_llm_model

        # Ensure model is available
        await self._model_manager.ensure_model(model)

        client = await self._get_client()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": self._build_options(temperature, top_p, top_k, num_predict, stop),
        }

        if system:
            payload["system"] = system

        async with client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        break

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_predict: int | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Generate chat completion.

        Args:
            messages: List of chat messages
            model: Model to use
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            num_predict: Max tokens to generate
            stop: Stop sequences
            stream: Whether to stream response

        Returns:
            Chat response
        """
        model = model or self.settings.default_llm_model

        # Ensure model is available
        await self._model_manager.ensure_model(model)

        client = await self._get_client()

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": self._build_options(temperature, top_p, top_k, num_predict, stop),
        }

        logger.debug("chat", model=model, message_count=len(messages))

        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()

        if stream:
            return {"streaming": True, "response": response}

        data = response.json()
        logger.debug(
            "chat_complete",
            model=model,
            response_length=len(data.get("message", {}).get("content", "")),
        )

        return data

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_predict: int | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion.

        Args:
            messages: List of chat messages
            model: Model to use
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            num_predict: Max tokens to generate
            stop: Stop sequences

        Yields:
            Generated text chunks
        """
        model = model or self.settings.default_llm_model

        # Ensure model is available
        await self._model_manager.ensure_model(model)

        client = await self._get_client()

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": self._build_options(temperature, top_p, top_k, num_predict, stop),
        }

        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                    if data.get("done"):
                        break


# Singleton instance
_completion_service: CompletionService | None = None


def get_completion_service() -> CompletionService:
    """Get completion service singleton."""
    global _completion_service
    if _completion_service is None:
        _completion_service = CompletionService()
    return _completion_service
