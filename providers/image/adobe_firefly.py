"""
Adobe Firefly Provider

Implementation for Adobe Firefly Image Generation.
Commercial-ready with legal security, free tier available.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import base64

from ..base_provider import BaseImageProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class AdobeFireflyProvider(BaseImageProvider):
    """
    Adobe Firefly Image Generation Provider.
    
    Features:
    - Commercial-ready with legal security
    - Adobe workflow integration
    - Legally secure for commercial projects
    - Free tier available
    
    Limitations:
    - Adobe ecosystem lock-in
    - May require Adobe account
    
    Best for: Professional designers, commercial projects
    """
    
    BASE_URL = "https://firefly-api.adobe.io/v1"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Adobe Firefly."""
        models = [
            ModelInfo(
                name="firefly-image-3",
                description="Adobe Firefly Image 3 - Commercial-ready",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["adobe", "commercial-ready", "legal-security"]
            ),
            ModelInfo(
                name="firefly-image-2",
                description="Adobe Firefly Image 2 - Previous version",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["adobe", "commercial-ready"]
            ),
        ]
        
        return ProviderConfig(
            name="Adobe Firefly",
            category=ProviderCategory.IMAGE,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="ADOBE_FIREFLY_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=10,  # Estimated
                requests_per_day=25,  # Estimated free tier
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
        Initialize Adobe Firefly provider.
        
        Args:
            api_key: Adobe Firefly API key
        """
        if api_key is None:
            api_key = os.getenv("ADOBE_FIREFLY_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Adobe Firefly API."""
        headers = super()._get_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["x-api-key"] = self.api_key
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
        Generate an image from a text prompt using Adobe Firefly.
        
        Args:
            prompt: Text description of the image
            model: Model to use (default: firefly-image-3)
            size: Image size (e.g., "1024x1024")
            quality: Quality setting
            **kwargs: Additional parameters
            
        Returns:
            bytes: Generated image data (PNG)
        """
        # Default model
        if model is None:
            model = "firefly-image-3"
        
        # Default size
        if size is None:
            size = "1024x1024"
        
        # Parse size
        try:
            width, height = map(int, size.lower().split('x'))
        except:
            width, height = 1024, 1024
        
        # Adobe Firefly payload
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "model": model,
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
        
        url = f"{self.BASE_URL}/images"
        
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
                    raise APIError(f"Adobe Firefly API error: {error_msg}")
                
                result = response.json()
                
                # Extract image data - Adobe Firefly may return a URL or base64
                if "image" in result:
                    image_data = result["image"]
                    if isinstance(image_data, str):
                        # Could be base64 or URL
                        if image_data.startswith("data:image"):
                            # Base64 encoded
                            return base64.b64decode(image_data.split(",")[1])
                        else:
                            # URL - need to download
                            return await self._download_image(image_data)
                
                raise APIError("No image data in response")
                
        except httpx.TimeoutException:
            raise APIError("Image generation timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def _download_image(self, url: str) -> bytes:
        """Download image from URL."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=60.0)
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