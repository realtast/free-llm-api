"""
Cohere Provider

Implementation for Cohere API.
Enterprise-focused with free tier: 20 req/min, 1,000 req/month.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseLLMProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class CohereProvider(BaseLLMProvider):
    """
    Cohere API Provider.
    
    Features:
    - Enterprise-grade models
    - Vision support
    - Monthly limit (not daily)
    - Shared quota across models
    
    Limitations:
    - Monthly limit (1,000 req/month)
    - Shared quota across all models
    
    Best for: Enterprise prototyping, vision applications
    """
    
    BASE_URL = "https://api.cohere.ai/v1"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Cohere."""
        models = [
            ModelInfo(
                name="c4ai-aya-expanse-32b",
                description="C4AI Aya Expanse 32B",
                parameters="32B",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["c4ai", "aya", "large"]
            ),
            ModelInfo(
                name="c4ai-aya-vision-32b",
                description="C4AI Aya Vision 32B",
                parameters="32B",
                context_window=128000,
                license="Proprietary",
                category="vision",
                tags=["c4ai", "aya", "vision"]
            ),
            ModelInfo(
                name="command-a-03-2025",
                description="Command A 03 2025",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["command", "latest"]
            ),
            ModelInfo(
                name="command-a-plus-05-2026",
                description="Command A Plus 05 2026",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["command", "plus", "latest"]
            ),
            ModelInfo(
                name="command-a-reasoning-08-2025",
                description="Command A Reasoning 08 2025",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="reasoning",
                tags=["command", "reasoning"]
            ),
            ModelInfo(
                name="command-a-translate-08-2025",
                description="Command A Translate 08 2025",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="translation",
                tags=["command", "translation"]
            ),
            ModelInfo(
                name="command-a-vision-07-2025",
                description="Command A Vision 07 2025",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="vision",
                tags=["command", "vision"]
            ),
            ModelInfo(
                name="command-r-08-2024",
                description="Command R 08 2024",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["command", "r-series"]
            ),
            ModelInfo(
                name="command-r-plus-08-2024",
                description="Command R Plus 08 2024",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["command", "r-series", "plus"]
            ),
            ModelInfo(
                name="command-r7b-12-2024",
                description="Command R7B 12 2024",
                parameters="7B",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["command", "r-series", "small"]
            ),
            ModelInfo(
                name="command-r7b-arabic-02-2025",
                description="Command R7B Arabic 02 2025",
                parameters="7B",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["command", "r-series", "arabic"]
            ),
        ]
        
        return ProviderConfig(
            name="Cohere",
            category=ProviderCategory.LLM,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="COHERE_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=20,
                requests_per_day=None,  # 1000 req/month, not daily
                tokens_per_day=None,
            ),
            requires_auth=True,
            requires_phone_verification=False,
            data_training_opt_out_available=True,
            commercial_usage_allowed=True,
            region="global",
            models=models
        )
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Cohere provider.
        
        Args:
            api_key: Cohere API key
        """
        if api_key is None:
            api_key = os.getenv("COHERE_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Cohere API."""
        headers = super()._get_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def chat(
        self,
        model: str,
        messages: Union[str, List[Dict[str, Any]]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send a chat completion request to Cohere.
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("Cohere rate limit exceeded")
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Cohere uses a different message format
        cohere_messages = []
        for msg in messages:
            if isinstance(msg, str):
                cohere_messages.append({"role": "USER", "message": msg})
            else:
                role = msg.get("role", "user").upper()
                cohere_messages.append({
                    "role": role,
                    "message": msg.get("content", "")
                })
        
        payload = {
            "model": model,
            "messages": cohere_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        # Add optional parameters
        if "p" in kwargs:
            payload["p"] = kwargs["p"]
        if "k" in kwargs:
            payload["k"] = kwargs["k"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]
        if "stop_sequences" in kwargs:
            payload["stop_sequences"] = kwargs["stop_sequences"]
        if "return_likelihoods" in kwargs:
            payload["return_likelihoods"] = kwargs["return_likelihoods"]
        
        url = f"{self.BASE_URL}/chat"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=120.0
                )
                
                self._update_request_counters()
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("message", "Unknown error")
                    raise APIError(f"Cohere API error: {error_msg}")
                
                result = response.json()
                
                # Extract text from response
                if "text" in result:
                    return result["text"]
                
                return ""
                
        except httpx.TimeoutException:
            raise APIError("Request timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def complete(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Send a text completion request."""
        # Cohere has a separate generate endpoint for completions
        if not self._check_rate_limits():
            raise RateLimitExceededError("Cohere rate limit exceeded")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        url = f"{self.BASE_URL}/generate"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=120.0
                )
                
                self._update_request_counters()
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("message", "Unknown error")
                    raise APIError(f"Cohere generate API error: {error_msg}")
                
                result = response.json()
                
                if "generations" in result and len(result["generations"]) > 0:
                    return result["generations"][0].get("text", "")
                
                return ""
                
        except httpx.TimeoutException:
            raise APIError("Request timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def embed(
        self,
        model: str,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        Generate embeddings using Cohere models.
        """
        url = f"{self.BASE_URL}/embed"
        
        payload = {
            "model": model,
            "texts": [text],
            "input_type": kwargs.get("input_type", "search_document")
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("message", "Unknown error")
                    raise APIError(f"Cohere embed API error: {error_msg}")
                
                result = response.json()
                
                if "embeddings" in result and len(result["embeddings"]) > 0:
                    return result["embeddings"][0]
                
                return []
                
        except httpx.TimeoutException:
            raise APIError("Request timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.session.aclose()


class APIError(Exception):
    """API error exception."""
    pass


class RateLimitExceededError(Exception):
    """Rate limit exceeded exception."""
    pass