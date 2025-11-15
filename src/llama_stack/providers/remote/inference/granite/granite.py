# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx  # pyright: ignore[reportMissingImports]
from llama_stack_api import (
    Inference,
    Model,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper

from .config import GraniteConfig

logger = get_logger(name=__name__, category="inference::granite")


class GraniteInferenceAdapter(
    ModelRegistryHelper,
    Inference,
):
    """IBM Granite Inference Adapter for Llama Stack."""

    __provider_id__: str

    def __init__(self, config: GraniteConfig) -> None:
        ModelRegistryHelper.__init__(self, allowed_models=config.allowed_models)
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._client_lock: asyncio.Lock | None = None

    async def initialize(self) -> None:
        """Initialize the HTTP client."""
        pass

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass
            finally:
                self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            if self._client_lock is None:
                self._client_lock = asyncio.Lock()

            async with self._client_lock:
                if self._client is None:
                    auth_header = ""
                    if self.config.auth_credential:
                        auth_header = f"Bearer {self.config.auth_credential.get_secret_value()}"

                    self._client = httpx.AsyncClient(
                        base_url=self.config.base_url.rstrip("/"),
                        headers={
                            "Authorization": auth_header,
                            "Content-Type": "application/json",
                        },
                        timeout=httpx.Timeout(60.0),
                        verify=False,
                        limits=httpx.Limits(
                            max_keepalive_connections=0,
                            max_connections=10,
                            keepalive_expiry=0.0,
                        ),
                    )
        return self._client

    async def check_model_availability(self, model: str) -> bool:
        """Check if a model is available from the Granite API."""
        try:
            models = await self.list_models()
            if models is None:
                return True
            return model in {m.provider_resource_id for m in models}
        except Exception:
            return True

    async def should_refresh_models(self) -> bool:
        """Whether to refresh models from the provider."""
        return self.config.refresh_models

    async def register_model(self, model: Model) -> Model:
        """Register a model with the provider."""
        return model

    async def unregister_model(self, model_id: str) -> None:
        """Unregister a model."""
        pass

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """Create embeddings using Granite API."""
        client = await self._get_client()
        # params.model is already the provider_resource_id (extracted by the router)
        provider_model_id = params.model

        request_data = {
            "model": provider_model_id,
            "input": params.input,
        }
        if params.encoding_format:
            request_data["encoding_format"] = params.encoding_format
        if params.dimensions:
            request_data["dimensions"] = params.dimensions
        if params.user:
            request_data["user"] = params.user

        response = await client.post("/v1/embeddings", json=request_data)
        response.raise_for_status()
        return OpenAIEmbeddingsResponse.model_validate(response.json())

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        """Create completion using Granite API."""
        client = await self._get_client()
        # params.model is already the provider_resource_id (extracted by the router)
        provider_model_id = params.model

        request_data = {
            "model": provider_model_id,
            "prompt": params.prompt,
        }
        if params.max_tokens:
            request_data["max_tokens"] = params.max_tokens
        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        if params.top_p is not None:
            request_data["top_p"] = params.top_p
        if params.frequency_penalty is not None:
            request_data["frequency_penalty"] = params.frequency_penalty
        if params.presence_penalty is not None:
            request_data["presence_penalty"] = params.presence_penalty
        if params.stop:
            request_data["stop"] = params.stop
        if params.stream is not None:
            request_data["stream"] = params.stream
        if params.user:
            request_data["user"] = params.user

        response = await client.post("/v1/completions", json=request_data)
        response.raise_for_status()
        return OpenAICompletion.model_validate(response.json())

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Create chat completion using Granite API."""
        client = await self._get_client()
        # params.model is already the provider_resource_id (extracted by the router)
        provider_model_id = params.model

        # Filter out None values from messages
        messages = [
            {k: v for k, v in msg.model_dump().items() if v is not None}
            for msg in params.messages
        ]

        request_data = {
            "model": provider_model_id,
            "messages": messages,
        }
        if params.max_tokens:
            request_data["max_tokens"] = params.max_tokens
        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        if params.top_p is not None:
            request_data["top_p"] = params.top_p
        if params.frequency_penalty is not None:
            request_data["frequency_penalty"] = params.frequency_penalty
        if params.presence_penalty is not None:
            request_data["presence_penalty"] = params.presence_penalty
        if params.stop:
            request_data["stop"] = params.stop
        if params.stream is not None:
            request_data["stream"] = params.stream
        if params.user:
            request_data["user"] = params.user

        if params.stream:
            async def stream_generator():
                async with client.stream("POST", "/v1/chat/completions", json=request_data) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                yield OpenAIChatCompletionChunk.model_validate(json.loads(data))
                            except Exception as e:
                                logger.warning(f"Failed to parse chunk: {e}")
                                continue
            return stream_generator()

        response = await client.post("/v1/chat/completions", json=request_data)
        response.raise_for_status()
        return OpenAIChatCompletion.model_validate(response.json())

    async def list_models(self) -> list[Model] | None:
        """List available models from Granite API."""
        try:
            client = await self._get_client()
            response = await client.get("/v1/models")
            response.raise_for_status()

            models_data = response.json()
            models = []

            if "data" in models_data:
                for model_data in models_data["data"]:
                    provider_model_id = model_data["id"]

                    if self.allowed_models and provider_model_id not in self.allowed_models:
                        continue

                    # Prefix identifier with provider ID for local registry (consistent with other providers)
                    local_identifier = f"{self.__provider_id__}/{provider_model_id}"
                    
                    models.append(
                        Model(
                            identifier=local_identifier,
                            provider_id=self.__provider_id__,
                            provider_resource_id=provider_model_id,
                            provider_model_id=provider_model_id,
                            name=model_data.get("name", provider_model_id),
                            description=model_data.get("description", ""),
                            context_length=model_data.get("context_length", 128000),
                            supports_chat=True,
                            supports_completion=True,
                            supports_embeddings=True,
                        )
                    )

            return models
        except Exception as e:
            logger.error(f"Failed to list models from Granite API: {e}")
            return None

    async def health_check(self) -> dict[str, Any]:
        """Check health of the Granite API."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            response.raise_for_status()
            try:
                return {"status": "healthy", "response": response.json()}
            except Exception:
                return {"status": "healthy", "response": response.text}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_version(self) -> dict[str, Any]:
        """Get version information from Granite API."""
        client = await self._get_client()
        response = await client.get("/version")
        response.raise_for_status()
        return response.json()

    async def tokenize(self, text: str) -> list[int]:
        """Tokenize text using Granite API."""
        client = await self._get_client()
        response = await client.post("/tokenize", json={"text": text})
        response.raise_for_status()
        return response.json().get("tokens", [])

    async def detokenize(self, tokens: list[int]) -> str:
        """Detokenize tokens using Granite API."""
        client = await self._get_client()
        response = await client.post("/detokenize", json={"tokens": tokens})
        response.raise_for_status()
        return response.json().get("text", "")
