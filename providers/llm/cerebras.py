"""
Cerebras Provider

Implementation for Cerebras API.
High-performance inference with very high daily limits: 14,400 req/day.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseLLMProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class CerebrasProvider(BaseLLMProvider):
    """
    Cerebras API Provider.
    
    Features:
    - Very high daily limits (14,400 req/day)
    - Strong performance
    - Limited model selection but high quality
    
    Limitations:
    - Limited model selection
    
    Best for: High-volume applications, research
    """
    
    BASE_URL = "https://api.cerebras.net/v1"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Cerebras."""
        models = [
            ModelInfo(
                name="gpt-oss-120b",
                description="GPT-OSS 120B",
                parameters="120B",
                context_window=128000,
                license="MIT",
                category="general",
                tags=["openai", "open-model", "high-volume"]
            ),
            ModelInfo(
                name="llama-3.1-8b",
                description="Llama 3.1 8B",
                parameters="8B",
                context_window=128000,
                license="Llama 3.1",
                category="general",
                tags=["llama", "high-volume"]
            ),
        ]
        
        return ProviderConfig(
            name="Cerebras",
            category=ProviderCategory.LLM,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="CEREBRAS_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=30,
                requests_per_hour=900,
                requests_per_day=14400,
                tokens_per_minute=60000,
                tokens_per_hour=1000000,
                tokens_per_day=1000000,
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
        Initialize Cerebras provider.
        
        Args:
            api_key: Cerebras API key
        """
        if api_key is None:
            api_key = os.getenv("CEREBRAS_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Cerebras API."""
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
        Send a chat completion request to Cerebras.
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("Cerebras rate limit exceeded")
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        cerebras_messages = []
        for msg in messages:
            if isinstance(msg, str):
                cerebras_messages.append({"role": "user", "content": msg})
            else:
                cerebras_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        payload = {
            "model": model,
            "messages": cerebras_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 8192,
        }
        
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
                    raise APIError(f"Cerebras API error: {error_msg}")
                
                result = response.json()
                
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