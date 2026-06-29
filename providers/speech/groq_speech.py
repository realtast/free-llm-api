"""
Groq Speech Provider

Implementation for Groq Speech API.
Best free speech-to-text with 2,000 req/day for Whisper Large v3.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import base64

from ..base_provider import BaseSpeechProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class GroqSpeechProvider(BaseSpeechProvider):
    """
    Groq Speech API Provider.
    
    Features:
    - High daily limit (2,000 req/day)
    - Production-ready
    - Fast inference
    - Whisper Large v3 models
    
    Limitations:
    - Speech-to-text only (no TTS in free tier)
    
    Best for: Speech-to-text applications, transcription
    """
    
    BASE_URL = "https://api.groq.com/v1"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Groq Speech."""
        models = [
            ModelInfo(
                name="whisper-large-v3",
                description="Whisper Large v3 - Production-ready STT",
                parameters="Unknown",
                context_window=None,
                license="MIT",
                category="speech",
                tags=["top-pick", "speech-to-text", "stt", "2000-req-day", "production-ready"]
            ),
            ModelInfo(
                name="whisper-large-v3-turbo",
                description="Whisper Large v3 Turbo - Faster STT",
                parameters="Unknown",
                context_window=None,
                license="MIT",
                category="speech",
                tags=["speech-to-text", "stt", "fast", "2000-req-day", "production-ready"]
            ),
        ]
        
        return ProviderConfig(
            name="Groq Speech",
            category=ProviderCategory.SPEECH,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="GROQ_API_KEY",
            rate_limits=RateLimit(
                requests_per_day=2000,
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
        Initialize Groq Speech provider.
        
        Args:
            api_key: Groq API key
        """
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
    
    async def transcribe(
        self,
        audio_data: Union[str, bytes],
        model: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio to text using Groq Speech.
        
        Args:
            audio_data: Path to audio file or audio data bytes
            model: Model to use (default: whisper-large-v3)
            language: Language hint
            **kwargs: Additional parameters
            
        Returns:
            str: Transcribed text
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("Groq Speech rate limit exceeded")
        
        # Default model
        if model is None:
            model = "whisper-large-v3"
        
        # Handle audio data
        if isinstance(audio_data, str):
            # It's a file path, read the file
            with open(audio_data, "rb") as f:
                audio_bytes = f.read()
        else:
            audio_bytes = audio_data
        
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Build payload
        payload = {
            "model": model,
            "audio": audio_base64,
        }
        
        # Add optional parameters
        if language:
            payload["language"] = language
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "prompt" in kwargs:
            payload["prompt"] = kwargs["prompt"]
        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]
        if "timestamp_granularities" in kwargs:
            payload["timestamp_granularities"] = kwargs["timestamp_granularities"]
        
        url = f"{self.BASE_URL}/audio/transcriptions"
        
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
                    raise APIError(f"Groq Speech API error: {error_msg}")
                
                result = response.json()
                
                # Extract text from response
                return result.get("text", "")
                
        except httpx.TimeoutException:
            raise APIError("Transcription timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def synthesize(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech.
        
        Note: Groq free tier is primarily for STT, not TTS.
        This method may not work with free tier.
        """
        raise NotImplementedError("Groq free tier does not support TTS")
    
    async def close(self):
        """Close the HTTP client."""
        await self.session.aclose()


class APIError(Exception):
    """API error exception."""
    pass


class RateLimitExceededError(Exception):
    """Rate limit exceeded exception."""
    pass