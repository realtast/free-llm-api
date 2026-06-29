"""
Vercel AI Gateway Provider

Implementation for Vercel AI Gateway.
AI gateway for Vercel applications with $5/month in credits.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseLLMProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class VercelProvider(BaseLLMProvider):
    """
    Vercel AI Gateway Provider.
    
    Features:
    - Easy integration with Vercel apps
    - Routes to various supported providers
    - $5/month in credits
    
    Limitations:
    - Limited to Vercel ecosystem
    - Credit-based system
    
    Best for: Vercel-hosted applications
    """
    
    BASE_URL = "https://api.vercel.com/v1/ai"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Vercel AI Gateway."""
        # Vercel routes to various providers, so we list the available models
        models = [
            ModelInfo(
                name="gpt-4o-mini",
                description="OpenAI GPT-4o Mini",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["openai", "gpt-4"]
            ),
            ModelInfo(
                name="gpt-4o",
                description="OpenAI GPT-4o",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["openai", "gpt-4"]
            ),
            ModelInfo(
                name="claude-3-5-sonnet",
                description="Anthropic Claude 3.5 Sonnet",
                parameters="Unknown",
                context_window=200000,
                license="Proprietary",
                category="general",
                tags=["anthropic", "claude"]
            ),
            ModelInfo(
                name="claude-3-haiku",
                description="Anthropic Claude 3 Haiku",
                parameters="Unknown",
                context_window=200000,
                license="Proprietary",
                category="general",
                tags=["anthropic", "claude"]
            ),
            ModelInfo(
                name="gemini-1.5-flash",
                description="Google Gemini 1.5 Flash",
                parameters="Unknown",
                context_window=1000000,
                license="Proprietary",
                category="general",
                tags=["google", "gemini"]
            ),
            ModelInfo(
                name="llama-3.1-70b-versatile",
                description="Llama 3.1 70B Versatile",
                parameters="70B",
                context_window=128000,
                license="Llama 3.1",
                category="general",
                tags=["llama", "meta"]
            ),
            ModelInfo(
                name="llama-3.1-8b-instant",
                description="Llama 3.1 8B Instant",
                parameters="8B",
                context_window=128000,
                license="Llama 3.1",
                category="general",
                tags=["llama", "meta"]
            ),
            ModelInfo(
                name="mistral-large",
                description="Mistral Large",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["mistral"]
            ),
            ModelInfo(
                name="mistral-small",
                description="Mistral Small",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["mistral"]
            ),
        ]
        
        return ProviderConfig(
            name="Vercel AI Gateway",
            category=ProviderCategory.LLM,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="VERCEL_AI_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=60,  # Estimated
                requests_per_day=100,  # Estimated based on $5 credits
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
        Initialize Vercel AI Gateway provider.
        
        Args:
            api_key: Vercel AI API key
        """
        if api_key is None:
            api_key = os.getenv("VERCEL_AI_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Vercel API."""
        headers = super()._get_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["Content-Type"] = "application/json"
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
        Send a chat completion request to Vercel AI Gateway.
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("Vercel AI Gateway rate limit exceeded")
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Vercel uses OpenAI-compatible format
        vercel_messages = []
        for msg in messages:
            if isinstance(msg, str):
                vercel_messages.append({"role": "user", "content": msg})
            else:
                vercel_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        payload = {
            "model": model,
            "messages": vercel_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 8192,
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "stream" in kwargs:
            payload["stream"] = kwargs["stream"]
        
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
                    raise APIError(f"Vercel AI API error: {error_msg}")
                
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