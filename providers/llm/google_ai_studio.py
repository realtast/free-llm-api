"""
Google AI Studio Provider

Implementation for Google AI Studio (formerly Google Vertex AI for free tier).
Offers the most generous free tier with up to 14,400 requests/day for Gemma 3 models.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseLLMProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class GoogleAIStudioProvider(BaseLLMProvider):
    """
    Google AI Studio API Provider.
    
    Features:
    - Most generous free tier: up to 14,400 requests/day for Gemma 3 models
    - 30 requests/minute rate limit
    - Multiple Gemma 3 models available
    - TTS models available
    - Google ecosystem integration
    
    Limitations:
    - Data may be used for training (outside UK/CH/EEA/EU)
    - Phone verification may be required
    - Rate limits vary by model
    
    Best for: Developers needing high-volume free access, prototyping, research
    """
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Google AI Studio."""
        models = [
            ModelInfo(
                name="gemma-3-27b-instruct",
                description="Gemma 3 27B Instruct - Top pick for free tier",
                parameters="27B",
                context_window=15000,
                license="Apache 2.0",
                category="general",
                tags=["top-pick", "high-volume", "apache-2.0"]
            ),
            ModelInfo(
                name="gemma-3-12b-instruct",
                description="Gemma 3 12B Instruct - Good for single GPU",
                parameters="12B",
                context_window=15000,
                license="Apache 2.0",
                category="general",
                tags=["single-gpu", "apache-2.0"]
            ),
            ModelInfo(
                name="gemma-3-4b-instruct",
                description="Gemma 3 4B Instruct - Laptop-friendly",
                parameters="4B",
                context_window=15000,
                license="Apache 2.0",
                category="general",
                tags=["laptop-friendly", "apache-2.0"]
            ),
            ModelInfo(
                name="gemma-3-1b-instruct",
                description="Gemma 3 1B Instruct - Ultra-lightweight",
                parameters="1B",
                context_window=15000,
                license="Apache 2.0",
                category="general",
                tags=["ultra-lightweight", "apache-2.0"]
            ),
            ModelInfo(
                name="gemini-3.5-flash",
                description="Gemini 3.5 Flash",
                parameters="Unknown",
                context_window=250000,
                license="Proprietary",
                category="general",
                tags=["high-context", "proprietary"]
            ),
            ModelInfo(
                name="gemini-3-flash",
                description="Gemini 3 Flash",
                parameters="Unknown",
                context_window=250000,
                license="Proprietary",
                category="general",
                tags=["high-context", "proprietary"]
            ),
            ModelInfo(
                name="gemini-3.1-flash-lite",
                description="Gemini 3.1 Flash-Lite - Best free tier",
                parameters="Unknown",
                context_window=250000,
                license="Proprietary",
                category="general",
                tags=["best-free-tier", "high-context", "proprietary"]
            ),
            ModelInfo(
                name="gemini-2.5-flash",
                description="Gemini 2.5 Flash - Legacy model",
                parameters="Unknown",
                context_window=250000,
                license="Proprietary",
                category="general",
                tags=["legacy", "high-context", "proprietary"]
            ),
            ModelInfo(
                name="gemini-2.5-flash-lite",
                description="Gemini 2.5 Flash-Lite - Lightweight option",
                parameters="Unknown",
                context_window=250000,
                license="Proprietary",
                category="general",
                tags=["lightweight", "high-context", "proprietary"]
            ),
            ModelInfo(
                name="gemma-3.1-flash-lite-tts",
                description="Gemma 3.1 Flash-Lite TTS - Text-to-speech",
                parameters="Unknown",
                context_window=10000,
                license="Proprietary",
                category="tts",
                tags=["text-to-speech", "tts"]
            ),
            ModelInfo(
                name="gemini-2.5-flash-tts",
                description="Gemini 2.5 Flash TTS - Text-to-speech",
                parameters="Unknown",
                context_window=10000,
                license="Proprietary",
                category="tts",
                tags=["text-to-speech", "tts"]
            ),
        ]
        
        return ProviderConfig(
            name="Google AI Studio",
            category=ProviderCategory.LLM,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="GOOGLE_AI_STUDIO_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=30,
                requests_per_day=14400,
                tokens_per_minute=15000,
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
        Initialize Google AI Studio provider.
        
        Args:
            api_key: Google AI Studio API key
        """
        # Try to get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_model_full_name(self, model: str) -> str:
        """Get the full model name for Google API."""
        model_mapping = {
            "gemma-3-27b-instruct": "models/gemma-3-27b-instruct",
            "gemma-3-12b-instruct": "models/gemma-3-12b-instruct", 
            "gemma-3-4b-instruct": "models/gemma-3-4b-instruct",
            "gemma-3-1b-instruct": "models/gemma-3-1b-instruct",
            "gemini-3.5-flash": "models/gemini-3.5-flash",
            "gemini-3-flash": "models/gemini-3-flash",
            "gemini-3.1-flash-lite": "models/gemini-3.1-flash-lite",
            "gemini-2.5-flash": "models/gemini-2.5-flash",
            "gemini-2.5-flash-lite": "models/gemini-2.5-flash-lite",
            "gemma-3.1-flash-lite-tts": "models/gemma-3.1-flash-lite-tts",
            "gemini-2.5-flash-tts": "models/gemini-2.5-flash-tts",
        }
        return model_mapping.get(model, f"models/{model}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Google AI Studio API."""
        headers = super()._get_headers()
        if self.api_key:
            headers["x-api-key"] = self.api_key
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
        Send a chat completion request to Google AI Studio.
        
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
            raise RateLimitExceededError("Google AI Studio rate limit exceeded")
        
        # Convert string message to list format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Format messages for Google API
        google_messages = []
        for msg in messages:
            if isinstance(msg, str):
                google_messages.append({"role": "user", "content": msg})
            else:
                google_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        payload = {
            "contents": google_messages,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens or 8192,
            }
        }
        
        # Add system instruction if provided
        if "system" in kwargs:
            payload["systemInstruction"] = kwargs["system"]
        
        model_full_name = self._get_model_full_name(model)
        url = f"{self.BASE_URL}/{model_full_name}:generateContent"
        
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
                    raise APIError(f"Google AI Studio API error: {error_msg}")
                
                result = response.json()
                
                # Extract text from response
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0].get("content", {})
                    if "parts" in content and len(content["parts"]) > 0:
                        return content["parts"][0].get("text", "")
                
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
        # For Google AI Studio, we'll use chat completion for text completion
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