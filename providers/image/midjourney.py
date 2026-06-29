"""
Midjourney Provider

Implementation for Midjourney API.
Best for artistic quality with free trial available.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import base64

from ..base_provider import BaseImageProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class MidjourneyProvider(BaseImageProvider):
    """
    Midjourney Image Generation Provider.
    
    Features:
    - Best for artistic and aesthetic quality
    - Unbeatable for artistic quality
    - Free trial available
    
    Limitations:
    - Paid plans for continued use
    - Slow generation (30-90 seconds)
    - Requires Discord integration
    
    Best for: Artistic projects, social media content
    """
    
    BASE_URL = "https://api.midjourney.com/v1"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Midjourney."""
        models = [
            ModelInfo(
                name="midjourney-v7",
                description="Midjourney v7 - Best for artistic quality",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["top-pick", "artistic", "high-quality"]
            ),
            ModelInfo(
                name="midjourney-v6",
                description="Midjourney v6 - Previous version",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["artistic", "high-quality"]
            ),
        ]
        
        return ProviderConfig(
            name="Midjourney",
            category=ProviderCategory.IMAGE,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="MIDJOURNEY_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=1,  # Very slow generation
                requests_per_day=10,  # Free trial limit
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
        Initialize Midjourney provider.
        
        Args:
            api_key: Midjourney API key
        """
        if api_key is None:
            api_key = os.getenv("MIDJOURNEY_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Midjourney API."""
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
        Generate an image from a text prompt using Midjourney.
        
        Args:
            prompt: Text description of the image
            model: Model to use (default: midjourney-v7)
            size: Image size (e.g., "1024x1024")
            quality: Quality setting
            **kwargs: Additional parameters
            
        Returns:
            bytes: Generated image data (PNG)
        """
        # Default model
        if model is None:
            model = "midjourney-v7"
        
        # Default size
        if size is None:
            size = "1024x1024"
        
        # Parse size
        try:
            width, height = map(int, size.lower().split('x'))
        except:
            width, height = 1024, 1024
        
        # Midjourney payload
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "model": model,
        }
        
        # Add optional parameters
        if "aspect_ratio" in kwargs:
            payload["aspect_ratio"] = kwargs["aspect_ratio"]
        if "chaos" in kwargs:
            payload["chaos"] = kwargs["chaos"]
        if "quality" in kwargs:
            payload["quality"] = kwargs["quality"]
        if "style" in kwargs:
            payload["style"] = kwargs["style"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]
        
        url = f"{self.BASE_URL}/images"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=300.0  # Midjourney can be slow
                )
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    raise APIError(f"Midjourney API error: {error_msg}")
                
                result = response.json()
                
                # Midjourney typically returns a URL to the generated image
                if "url" in result:
                    return await self._download_image(result["url"])
                elif "image_url" in result:
                    return await self._download_image(result["image_url"])
                elif "image" in result:
                    image_data = result["image"]
                    if isinstance(image_data, str):
                        if image_data.startswith("data:image"):
                            return base64.b64decode(image_data.split(",")[1])
                        else:
                            return await self._download_image(image_data)
                
                raise APIError("No image data in response")
                
        except httpx.TimeoutException:
            raise APIError("Image generation timeout - Midjourney can take 30-90 seconds")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def _download_image(self, url: str) -> bytes:
        """Download image from URL."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=120.0)
                if response.status_code == 200:
                    return response.content
                else:
                    raise APIError(f"Failed to download image: {response.status_code}")
        except Exception as e:
            raise APIError(f"Image download failed: {str(e)}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.session.aclose()


class APIError(Exception):
    """API error exception."""
    pass