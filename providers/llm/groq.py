"""
Groq Provider

Implementation for Groq API.
Offers extremely fast inference with generous limits: 14,400 req/day for Llama 3.1 8B.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseLLMProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class GroqProvider(BaseLLMProvider):
    """
    Groq API Provider.
    
    Features:
    - Extremely fast inference
    - High daily limits (14,400 req/day for Llama 3.1 8B)
    - Production-ready
    - Multiple open models available
    - Speech-to-text models (Whisper Large v3)
    
    Limitations:
    - Some models have lower daily caps
    - Rate limits vary by model
    
    Best for: Production applications, speed-critical use cases, bulk processing
    """
    
    BASE_URL = "https://api.groq.com/v1"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Groq."""
        models = [
            ModelInfo(
                name="llama-3.1-8b-instant",
                description="Llama 3.1 8B - Best overall with 14,400 req/day",
                parameters="8B",
                context_window=131072,
                license="Llama 3.1",
                category="general",
                tags=["top-pick", "high-volume", "fast"]
            ),
            ModelInfo(
                name="allam-2-7b",
                description="Allam 2 7B - 7,000 req/day",
                parameters="7B",
                context_window=131072,
                license="Allam 2",
                category="general",
                tags=["high-volume", "fast"]
            ),
            ModelInfo(
                name="llama-3.3-70b-versatile",
                description="Llama 3.3 70B - Higher capacity",
                parameters="70B",
                context_window=131072,
                license="Llama 3.3",
                category="general",
                tags=["high-capacity", "fast"]
            ),
            ModelInfo(
                name="llama-4-scout-instruct",
                description="Llama 4 Scout Instruct - Long context",
                parameters="17B",
                context_window=10000000,
                license="Llama 4",
                category="general",
                tags=["long-context", "fast"]
            ),
            ModelInfo(
                name="whisper-large-v3",
                description="Whisper Large v3 - Speech-to-text",
                parameters="Unknown",
                context_window=None,
                license="MIT",
                category="speech",
                tags=["speech-to-text", "stt", "2000-req-day"]
            ),
            ModelInfo(
                name="whisper-large-v3-turbo",
                description="Whisper Large v3 Turbo - Faster STT",
                parameters="Unknown",
                context_window=None,
                license="MIT",
                category="speech",
                tags=["speech-to-text", "stt", "fast", "2000-req-day"]
            ),
            ModelInfo(
                name="openai/gpt-oss-120b",
                description="OpenAI's open model GPT-OSS 120B",
                parameters="120B",
                context_window=131072,
                license="MIT",
                category="general",
                tags=["openai", "open-model", "1000-req-day"]
            ),
            ModelInfo(
                name="openai/gpt-oss-20b",
                description="OpenAI's open model GPT-OSS 20B",
                parameters="20B",
                context_window=131072,
                license="MIT",
                category="general",
                tags=["openai", "open-model", "1000-req-day"]
            ),
            ModelInfo(
                name="qwen/qwen3-32b",
                description="Qwen 3 32B - Strong reasoning",
                parameters="32B",
                context_window=131072,
                license="Apache 2.0",
                category="general",
                tags=["qwen", "reasoning", "1000-req-day"]
            ),
            ModelInfo(
                name="qwen/qwen3.6-27b",
                description="Qwen 3.6 27B - Latest Qwen",
                parameters="27B",
                context_window=131072,
                license="Apache 2.0",
                category="general",
                tags=["qwen", "latest", "1000-req-day"]
            ),
            ModelInfo(
                name="groq/compound",
                description="Groq Compound - Specialized",
                parameters="Unknown",
                context_window=70000,
                license="Proprietary",
                category="specialized",
                tags=["specialized", "250-req-day"]
            ),
            ModelInfo(
                name="groq/compound-mini",
                description="Groq Compound Mini - Lightweight",
                parameters="Unknown",
                context_window=70000,
                license="Proprietary",
                category="specialized",
                tags=["lightweight", "250-req-day"]
            ),
        ]
        
        return ProviderConfig(
            name="Groq",
            category=ProviderCategory.LLM,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="GROQ_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=6000,  # tokens per minute
                requests_per_day=14400,  # for Llama 3.1 8B
                tokens_per_minute=6000,
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
        Initialize Groq provider.
        
        Args:
            api_key: Groq API key
        """
        # Try to get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Groq API."""
        headers = super()._get_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _get_model_rate_limits(self, model: str) -> RateLimit:
        """Get rate limits for a specific model."""
        model_limits = {
            "llama-3.1-8b-instant": RateLimit(
                requests_per_day=14400,
                tokens_per_minute=6000
            ),
            "allam-2-7b": RateLimit(
                requests_per_day=7000,
                tokens_per_minute=6000
            ),
            "llama-3.3-70b-versatile": RateLimit(
                requests_per_day=1000,
                tokens_per_minute=12000
            ),
            "llama-4-scout-instruct": RateLimit(
                requests_per_day=1000,
                tokens_per_minute=30000
            ),
            "whisper-large-v3": RateLimit(
                requests_per_day=2000
            ),
            "whisper-large-v3-turbo": RateLimit(
                requests_per_day=2000
            ),
            "openai/gpt-oss-120b": RateLimit(
                requests_per_day=1000,
                tokens_per_minute=8000
            ),
            "openai/gpt-oss-20b": RateLimit(
                requests_per_day=1000,
                tokens_per_minute=8000
            ),
            "qwen/qwen3-32b": RateLimit(
                requests_per_day=1000,
                tokens_per_minute=6000
            ),
            "qwen/qwen3.6-27b": RateLimit(
                requests_per_day=1000,
                tokens_per_minute=8000
            ),
            "groq/compound": RateLimit(
                requests_per_day=250,
                tokens_per_minute=70000
            ),
            "groq/compound-mini": RateLimit(
                requests_per_day=250,
                tokens_per_minute=70000
            ),
        }
        return model_limits.get(model, self.config.rate_limits)
    
    async def chat(
        self,
        model: str,
        messages: Union[str, List[Dict[str, Any]]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send a chat completion request to Groq.
        
        Args:
            model: Model name
            messages: Chat messages (string or list of message dicts)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            str: Generated text response
        """
        # Get model-specific rate limits
        model_limits = self._get_model_rate_limits(model)
        
        # Temporarily override rate limits for this request
        original_limits = self.config.rate_limits
        self.config.rate_limits = model_limits
        
        try:
            if not self._check_rate_limits():
                raise RateLimitExceededError(f"Groq rate limit exceeded for model {model}")
            
            # Convert string message to list format
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            # Format messages for Groq API (OpenAI-compatible format)
            groq_messages = []
            for msg in messages:
                if isinstance(msg, str):
                    groq_messages.append({"role": "user", "content": msg})
                else:
                    groq_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            payload = {
                "model": model,
                "messages": groq_messages,
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
                    raise APIError(f"Groq API error: {error_msg}")
                
                result = response.json()
                
                # Extract text from response
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0].get("message", {}).get("content", "")
                
                return ""
                
        except httpx.TimeoutException:
            raise APIError("Request timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
        finally:
            # Restore original rate limits
            self.config.rate_limits = original_limits
    
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
        # For Groq, we'll use chat completion for text completion
        return await self.chat(model, prompt, temperature, max_tokens, **kwargs)
    
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