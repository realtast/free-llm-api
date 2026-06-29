"""
Google Gemini Image Provider

Implementation for Google Gemini Image Generation.
Best free option for most users with low setup effort.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import base64

from ..base_provider import BaseImageProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class GoogleGeminiImageProvider(BaseImageProvider):
    """
    Google Gemini Image Generation Provider.
    
    Features:
    - Best free option for most users
    - Low setup effort
    - Browser-based
    - Broad use case support
    
    Limitations:
    - Limited free tier
    - Less control than local options
    
    Best for: Beginners, quick generation, broad use cases
    """
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Google Gemini Image."""
        models = [
            ModelInfo(
                name="gemini-1.5-flash-image",
                description="Gemini 1.5 Flash Image - Best for most users",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["top-pick", "google", "easy-to-use"]
            ),
            ModelInfo(
                name="gemini-1.5-pro-image",
                description="Gemini 1.5 Pro Image - Higher quality",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["google", "high-quality"]
            ),
            ModelInfo(
                name="imagen-3",
                description="Imagen 3 - Best for text rendering",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["google", "text-rendering"]
            ),
        ]
        
        return ProviderConfig(
            name="Google Gemini Image",
            category=ProviderCategory.IMAGE,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="GOOGLE_GEMINI_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=10,  # Estimated
                requests_per_day=50,  # Estimated free tier
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
        Initialize Google Gemini Image provider.
        
        Args:
            api_key: Google Gemini API key
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Google Gemini API."""
        headers = super()._get_headers()
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
    
    def _get_model_full_name(self, model: str) -> str:
        """Get the full model name for Google API."""
        model_mapping = {
            "gemini-1.5-flash-image": "models/gemini-1.5-flash-image",
            "gemini-1.5-pro-image": "models/gemini-1.5-pro-image",
            "imagen-3": "models/imagen-3",
        }
        return model_mapping.get(model, f"models/{model}")
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Generate an image from a text prompt using Google Gemini.
        
        Args:
            prompt: Text description of the image
            model: Model to use (default: gemini-1.5-flash-image)
            size: Image size (e.g., "1024x1024")
            quality: Quality setting
            **kwargs: Additional parameters
            
        Returns:
            bytes: Generated image data (PNG)
        """
        # Default model
        if model is None:
            model = "gemini-1.5-flash-image"
        
        # Default size
        if size is None:
            size = "1024x1024"
        
        # Parse size
        try:
            width, height = map(int, size.lower().split('x'))
        except:
            width, height = 1024, 1024
        
        # Google Gemini image generation payload
        payload = {
            "prompt": prompt,
            "image_size": {
                "width": width,
                "height": height
            }
        }
        
        # Add optional parameters
        if "negative_prompt" in kwargs:
            payload["negative_prompt"] = kwargs["negative_prompt"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]
        if "steps" in kwargs:
            payload["steps"] = kwargs["steps"]
        if "cfg_scale" in kwargs:
            payload["cfg_scale"] = kwargs["cfg_scale"]
        
        model_full_name = self._get_model_full_name(model)
        url = f"{self.BASE_URL}/{model_full_name}:generateImage"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=120.0
                )
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    raise APIError(f"Google Gemini Image API error: {error_msg}")
                
                result = response.json()
                
                # Extract image data
                if "image" in result:
                    # Base64 encoded image data
                    image_data = result["image"]
                    return base64.b64decode(image_data)
                
                raise APIError("No image data in response")
                
        except httpx.TimeoutException:
            raise APIError("Image generation timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.session.aclose()


class APIError(Exception):
    """API error exception."""
    pass