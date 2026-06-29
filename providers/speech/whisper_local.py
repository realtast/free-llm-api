"""
Whisper Local Provider

Implementation for local Whisper deployment.
Completely free, no limits, full control, privacy.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import base64

from ..base_provider import BaseSpeechProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class WhisperLocalProvider(BaseSpeechProvider):
    """
    Local Whisper Provider.
    
    Features:
    - Completely free
    - No limits (hardware-limited only)
    - Full control
    - Privacy (no data sent to cloud)
    - Offline capability
    
    Limitations:
    - Hardware requirements
    - Requires local setup
    
    Best for: Offline STT, privacy-focused applications
    """
    
    BASE_URL = "http://localhost:9000"  # Default for local Whisper API
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for local Whisper."""
        models = [
            ModelInfo(
                name="whisper-1",
                description="Whisper v1 - Original version",
                parameters="1550M",
                context_window=None,
                license="MIT",
                category="speech",
                tags=["top-pick", "speech-to-text", "stt", "local", "offline", "privacy"]
            ),
            ModelInfo(
                name="whisper-large-v3",
                description="Whisper Large v3 - Latest version",
                parameters="1550M",
                context_window=None,
                license="MIT",
                category="speech",
                tags=["speech-to-text", "stt", "local", "offline", "privacy", "latest"]
            ),
            ModelInfo(
                name="whisper-large-v3-turbo",
                description="Whisper Large v3 Turbo - Faster version",
                parameters="1550M",
                context_window=None,
                license="MIT",
                category="speech",
                tags=["speech-to-text", "stt", "local", "offline", "privacy", "fast"]
            ),
        ]
        
        return ProviderConfig(
            name="Whisper Local",
            category=ProviderCategory.SPEECH,
            provider_type=ProviderType.LOCAL,
            base_url=cls.BASE_URL,
            api_key_env_var=None,  # No API key needed for local
            rate_limits=RateLimit(
                requests_per_minute=None,  # Hardware-limited only
                requests_per_day=None,
            ),
            requires_auth=False,
            requires_phone_verification=False,
            data_training_opt_out_available=True,
            commercial_usage_allowed=True,
            region="local",
            models=models
        )
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize Whisper Local provider.
        
        Args:
            base_url: Base URL for local Whisper API
        """
        if base_url is None:
            base_url = os.getenv("WHISPER_BASE_URL", self.BASE_URL)
        
        # Create a custom config with the provided base URL
        config = self.get_default_config()
        config.base_url = base_url
        
        super().__init__(config=config)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Whisper API."""
        headers = super()._get_headers()
        headers["Content-Type"] = "application/json"
        return headers
    
    async def transcribe(
        self,
        audio_data: Union[str, bytes],
        model: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio to text using local Whisper.
        
        Args:
            audio_data: Path to audio file or audio data bytes
            model: Model to use (default: whisper-large-v3)
            language: Language hint
            **kwargs: Additional parameters
            
        Returns:
            str: Transcribed text
        """
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
        
        # Build payload for local Whisper API
        payload = {
            "audio": audio_base64,
            "model": model,
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
        
        url = f"{self.config.base_url}/transcribe"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=300.0  # Transcription can take time
                )
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Unknown error")
                    raise APIError(f"Whisper Local API error: {error_msg}")
                
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
        
        Note: Whisper is for speech-to-text only, not text-to-speech.
        """
        raise NotImplementedError("Whisper is for speech-to-text only, not text-to-speech")
    
    async def close(self):
        """Close the HTTP client."""
        await self.session.aclose()


class APIError(Exception):
    """API error exception."""
    pass