"""
Imagen Provider

Implementation for Google Imagen 3.
Best for text rendering in images.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import base64

from ..base_provider import BaseImageProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class ImagenProvider(BaseImageProvider):
    """
    Google Imagen 3 Image Generation Provider.
    
    Features:
    - Best at rendering text in images
    - Excellent for product mockups, social media graphics, banners
    - High-quality text rendering
    
    Limitations:
    - May require Google Cloud account
    - Free tier details unclear
    
    Best for: Images requiring readable text, product mockups, social media graphics
    """
    
    BASE_URL = "https://imagen.googleapis.com/v1"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Imagen."""
        models = [
            ModelInfo(
                name="imagen-3",
                description="Imagen 3 - Best for text rendering",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["top-pick", "google", "text-rendering", "product-mockups"]
            ),
            ModelInfo(
                name="imagen-2",
                description="Imagen 2 - Previous version",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["google", "text-rendering"]
            ),
        ]
        
        return ProviderConfig(
            name="Imagen",
            category=ProviderCategory.IMAGE,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="IMAGEN_API_KEY",
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
        Initialize Imagen provider.
        
        Args:
            api_key: Imagen API key
        """
        if api_key is None:
            api_key = os.getenv("IMAGEN_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Imagen API."""
        headers = super()._get_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Generate an image from a text prompt using Imagen.
        
        Args:
            prompt: Text description of the image
            model: Model to use (default: imagen-3)
            size: Image size (e.g., "1024x1024")
            quality: Quality setting
            **kwargs: Additional parameters
            
        Returns:
            bytes: Generated image data (PNG)
        """
        # Default model
        if model is None:
            model = "imagen-3"
        
        # Default size
        if size is None:
            size = "1024x1024"
        
        # Parse size
        try:
            width, height = map(int, size.lower().split('x'))
        except:
            width, height = 1024, 1024
        
        # Imagen payload
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
        
        # Imagen uses model in the URL path
        url = f"{self.BASE_URL}/projects/{model}:generate"
        
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
                    raise APIError(f"Imagen API error: {error_msg}")
                
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