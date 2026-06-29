"""
Google TTS Provider

Implementation for Google AI Studio TTS.
Text-to-speech with 10 requests/day, 3 requests/minute.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import base64

from ..base_provider import BaseSpeechProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class GoogleTTSProvider(BaseSpeechProvider):
    """
    Google AI Studio TTS Provider.
    
    Features:
    - Google-quality TTS
    - Free tier available
    - Multiple TTS models
    
    Limitations:
    - Low daily limit (10 requests/day)
    - Low rate limit (3 requests/minute)
    
    Best for: Text-to-speech applications
    """
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Google TTS."""
        models = [
            ModelInfo(
                name="gemma-3.1-flash-lite-tts",
                description="Gemma 3.1 Flash-Lite TTS",
                parameters="Unknown",
                context_window=10000,
                license="Proprietary",
                category="tts",
                tags=["google", "text-to-speech", "tts", "10-req-day"]
            ),
            ModelInfo(
                name="gemini-2.5-flash-tts",
                description="Gemini 2.5 Flash TTS",
                parameters="Unknown",
                context_window=10000,
                license="Proprietary",
                category="tts",
                tags=["google", "text-to-speech", "tts", "10-req-day"]
            ),
        ]
        
        return ProviderConfig(
            name="Google TTS",
            category=ProviderCategory.SPEECH,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="GOOGLE_AI_STUDIO_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=3,
                requests_per_day=10,
                tokens_per_minute=10000,
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
        Initialize Google TTS provider.
        
        Args:
            api_key: Google AI Studio API key
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Google API."""
        headers = super()._get_headers()
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
    
    def _get_model_full_name(self, model: str) -> str:
        """Get the full model name for Google API."""
        model_mapping = {
            "gemma-3.1-flash-lite-tts": "models/gemma-3.1-flash-lite-tts",
            "gemini-2.5-flash-tts": "models/gemini-2.5-flash-tts",
        }
        return model_mapping.get(model, f"models/{model}")
    
    async def transcribe(
        self,
        audio_data: Union[str, bytes],
        model: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio to text.
        
        Note: Google TTS is primarily for text-to-speech, not speech-to-text.
        This method may not be available.
        """
        raise NotImplementedError("Google TTS provider does not support speech-to-text")
    
    async def synthesize(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech using Google TTS.
        
        Args:
            text: Text to synthesize
            model: Model to use (default: gemma-3.1-flash-lite-tts)
            voice: Voice to use
            **kwargs: Additional parameters
            
        Returns:
            bytes: Audio data (WAV or MP3)
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("Google TTS rate limit exceeded")
        
        # Default model
        if model is None:
            model = "gemma-3.1-flash-lite-tts"
        
        # Build payload
        payload = {
            "input": text,
        }
        
        # Add optional parameters
        if voice:
            payload["voice"] = voice
        if "speed" in kwargs:
            payload["speed"] = kwargs["speed"]
        if "pitch" in kwargs:
            payload["pitch"] = kwargs["pitch"]
        if "audio_format" in kwargs:
            payload["audio_format"] = kwargs["audio_format"]
        
        model_full_name = self._get_model_full_name(model)
        url = f"{self.BASE_URL}/{model_full_name}:generateSpeech"
        
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
                    raise APIError(f"Google TTS API error: {error_msg}")
                
                result = response.json()
                
                # Extract audio data
                if "audio" in result:
                    # Base64 encoded audio data
                    audio_data = result["audio"]
                    return base64.b64decode(audio_data)
                
                raise APIError("No audio data in response")
                
        except httpx.TimeoutException:
            raise APIError("Synthesis timeout")
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