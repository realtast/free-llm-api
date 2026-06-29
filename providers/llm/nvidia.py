"""
NVIDIA NIM Provider

Implementation for NVIDIA NIM API.
Optimized inference for NVIDIA GPUs with 40 req/min.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseLLMProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class NVIDIAProvider(BaseLLMProvider):
    """
    NVIDIA NIM API Provider.
    
    Features:
    - NVIDIA-optimized inference
    - Good for NVIDIA hardware users
    - 40 requests per minute
    
    Limitations:
    - Phone verification required
    - Limited context window
    - NVIDIA ecosystem lock-in
    
    Best for: NVIDIA GPU users, optimized inference
    """
    
    BASE_URL = "https://api.nvidia.com/v1"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for NVIDIA NIM."""
        models = [
            ModelInfo(
                name="meta/llama3-1-8b-instruct",
                description="Llama 3.1 8B Instruct",
                parameters="8B",
                context_window=128000,
                license="Llama 3.1",
                category="general",
                tags=["llama", "meta", "nvidia-optimized"]
            ),
            ModelInfo(
                name="meta/llama3-2-11b-vision-instruct",
                description="Llama 3.2 11B Vision Instruct",
                parameters="11B",
                context_window=128000,
                license="Llama 3.2",
                category="vision",
                tags=["llama", "meta", "vision", "nvidia-optimized"]
            ),
            ModelInfo(
                name="mistralai/mistral-7b-instruct-v0.3",
                description="Mistral 7B Instruct v0.3",
                parameters="7B",
                context_window=32768,
                license="Apache 2.0",
                category="general",
                tags=["mistral", "apache-2.0", "nvidia-optimized"]
            ),
            ModelInfo(
                name="mistralai/mixtral-8x7b-instruct-v0.1",
                description="Mixtral 8x7B Instruct v0.1",
                parameters="47B",
                context_window=32768,
                license="Apache 2.0",
                category="general",
                tags=["mistral", "mixtral", "apache-2.0", "nvidia-optimized"]
            ),
            ModelInfo(
                name="qwen/qwen2.5-7b-instruct",
                description="Qwen 2.5 7B Instruct",
                parameters="7B",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["qwen", "apache-2.0", "nvidia-optimized"]
            ),
            ModelInfo(
                name="qwen/qwen2.5-coder-7b-instruct",
                description="Qwen 2.5 Coder 7B Instruct",
                parameters="7B",
                context_window=128000,
                license="Apache 2.0",
                category="coding",
                tags=["qwen", "coding", "apache-2.0", "nvidia-optimized"]
            ),
            ModelInfo(
                name="google/gemma-2-9b-it",
                description="Gemma 2 9B IT",
                parameters="9B",
                context_window=8192,
                license="Apache 2.0",
                category="general",
                tags=["google", "gemma", "apache-2.0", "nvidia-optimized"]
            ),
            ModelInfo(
                name="google/gemma-2-27b-it",
                description="Gemma 2 27B IT",
                parameters="27B",
                context_window=8192,
                license="Apache 2.0",
                category="general",
                tags=["google", "gemma", "apache-2.0", "nvidia-optimized"]
            ),
        ]
        
        return ProviderConfig(
            name="NVIDIA NIM",
            category=ProviderCategory.LLM,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="NVIDIA_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=40,
            ),
            requires_auth=True,
            requires_phone_verification=True,
            data_training_opt_out_available=True,
            commercial_usage_allowed=True,
            region="global",
            models=models
        )
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NVIDIA NIM provider.
        
        Args:
            api_key: NVIDIA API key
        """
        if api_key is None:
            api_key = os.getenv("NVIDIA_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for NVIDIA API."""
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
        Send a chat completion request to NVIDIA NIM.
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("NVIDIA NIM rate limit exceeded")
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # NVIDIA uses OpenAI-compatible format
        nvidia_messages = []
        for msg in messages:
            if isinstance(msg, str):
                nvidia_messages.append({"role": "user", "content": msg})
            else:
                nvidia_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        payload = {
            "model": model,
            "messages": nvidia_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 8192,
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        
        url = f"{self.BASE_URL}/chat/completions"
        
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
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    raise APIError(f"NVIDIA API error: {error_msg}")
                
                result = response.json()
                
                # Extract text from response
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0].get("message", {}).get("content", "")
                
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
        return await self.chat(model, prompt, temperature, max_tokens, **kwargs)
    
    async def close(self):
        """Close the HTTP client."""
        await self.session.aclose()


class APIError(Exception):
    """API error exception."""
    pass


class RateLimitExceededError(Exception):
    """Rate limit exceeded exception."""
    pass