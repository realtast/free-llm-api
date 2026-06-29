"""
Mistral AI Provider

Implementation for Mistral AI API.
European provider with strong open-source focus, offering 1M tokens/month free.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseLLMProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class MistralProvider(BaseLLMProvider):
    """
    Mistral AI API Provider.
    
    Features:
    - Massive monthly token allowance (1M tokens/month)
    - Enterprise-grade infrastructure
    - Full Mistral model family (open and proprietary)
    - Strong open-source focus
    
    Limitations:
    - Phone verification required
    - Data training opt-in required
    - Rate limits: 1 req/sec, 500,000 tokens/min
    
    Best for: Enterprise prototyping, European users, open-source enthusiasts
    """
    
    BASE_URL = "https://api.mistral.ai/v1"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Mistral AI."""
        models = [
            ModelInfo(
                name="mistral-tiny-2402",
                description="Mistral Tiny - Fast and efficient",
                parameters="Unknown",
                context_window=32768,
                license="Apache 2.0",
                category="general",
                tags=["fast", "efficient", "apache-2.0"]
            ),
            ModelInfo(
                name="mistral-small-2402",
                description="Mistral Small - Balanced performance",
                parameters="24B",
                context_window=32768,
                license="Apache 2.0",
                category="general",
                tags=["balanced", "apache-2.0"]
            ),
            ModelInfo(
                name="mistral-medium-2402",
                description="Mistral Medium - High performance",
                parameters="Unknown",
                context_window=32768,
                license="Proprietary",
                category="general",
                tags=["high-performance", "proprietary"]
            ),
            ModelInfo(
                name="mistral-large-2402",
                description="Mistral Large - State-of-the-art",
                parameters="Unknown",
                context_window=32768,
                license="Proprietary",
                category="general",
                tags=["state-of-the-art", "proprietary"]
            ),
            ModelInfo(
                name="mistral-small-3.1-24b",
                description="Mistral Small 3.1 24B",
                parameters="24B",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["apache-2.0", "long-context"]
            ),
            ModelInfo(
                name="mistral-medium-3.1",
                description="Mistral Medium 3.1",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["proprietary", "long-context"]
            ),
            ModelInfo(
                name="codestral-2405",
                description="Codestral - Specialized for coding",
                parameters="24B",
                context_window=32768,
                license="Apache 2.0",
                category="coding",
                tags=["coding", "apache-2.0"]
            ),
            ModelInfo(
                name="ministral-3b",
                description="Ministral 3B - Lightweight",
                parameters="3B",
                context_window=32768,
                license="Apache 2.0",
                category="general",
                tags=["lightweight", "apache-2.0"]
            ),
        ]
        
        return ProviderConfig(
            name="Mistral AI",
            category=ProviderCategory.LLM,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="MISTRAL_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=1,  # 1 req/sec = 60 req/min
                tokens_per_minute=500000,
                tokens_per_day=1000000,  # 1M tokens/month
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
        Initialize Mistral AI provider.
        
        Args:
            api_key: Mistral API key
        """
        # Try to get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Mistral API."""
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
        Send a chat completion request to Mistral AI.
        
        Args:
            model: Model name
            messages: Chat messages (string or list of message dicts)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            str: Generated text response
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("Mistral AI rate limit exceeded")
        
        # Convert string message to list format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Format messages for Mistral API (OpenAI-compatible format)
        mistral_messages = []
        for msg in messages:
            if isinstance(msg, str):
                mistral_messages.append({"role": "user", "content": msg})
            else:
                mistral_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        payload = {
            "model": model,
            "messages": mistral_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 8192,
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "random_seed" in kwargs:
            payload["random_seed"] = kwargs["random_seed"]
        if "safe_mode" in kwargs:
            payload["safe_mode"] = kwargs["safe_mode"]
        
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
                    raise APIError(f"Mistral API error: {error_msg}")
                
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
        """
        Send a text completion request.
        
        Args:
            model: Model name
            prompt: Text prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            str: Generated text
        """
        # For Mistral, we'll use chat completion for text completion
        return await self.chat(model, prompt, temperature, max_tokens, **kwargs)
    
    async def embed(
        self,
        model: str,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        Generate embeddings using Mistral models.
        
        Args:
            model: Embedding model name
            text: Text to embed
            **kwargs: Additional parameters
            
        Returns:
            List[float]: Embedding vector
        """
        # Mistral also offers embedding models
        url = f"{self.BASE_URL}/embeddings"
        
        payload = {
            "model": model,
            "input": text
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
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    raise APIError(f"Mistral embedding API error: {error_msg}")
                
                result = response.json()
                
                # Extract embedding from response
                if "data" in result and len(result["data"]) > 0:
                    return result["data"][0].get("embedding", [])
                
                return []
                
        except httpx.TimeoutException:
            raise APIError("Request timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.session.aclose()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self.close())


class APIError(Exception):
    """API error exception."""
    pass


class RateLimitExceededError(Exception):
    """Rate limit exceeded exception."""
    pass