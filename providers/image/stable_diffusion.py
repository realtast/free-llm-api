"""
Stable Diffusion Provider

Implementation for Stable Diffusion.
Completely free for local deployment with no platform caps.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import base64
from io import BytesIO

from ..base_provider import BaseImageProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class StableDiffusionProvider(BaseImageProvider):
    """
    Stable Diffusion Provider.
    
    Features:
    - Completely free (local deployment)
    - High-quality results
    - Technically customizable
    - Commercial use allowed
    - No platform caps
    
    Limitations:
    - Hardware-limited
    - Requires setup
    - Large version requires ~18GB FP16 (FP8 quantization reduces this)
    
    Best for: Developers, technical users, commercial projects
    """
    
    BASE_URL = "http://localhost:7860"  # Default for local ComfyUI
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Stable Diffusion."""
        models = [
            ModelInfo(
                name="stable-diffusion-3.5",
                description="Stable Diffusion 3.5 - Most flexible free option",
                parameters="2B",
                context_window=None,
                license="CreativeML OpenRAIL-M",
                category="image",
                tags=["top-pick", "commercial-allowed", "local"]
            ),
            ModelInfo(
                name="stable-diffusion-xl",
                description="Stable Diffusion XL - High quality",
                parameters="3.5B",
                context_window=None,
                license="CreativeML OpenRAIL-M",
                category="image",
                tags=["high-quality", "commercial-allowed", "local"]
            ),
            ModelInfo(
                name="stable-diffusion-2.1",
                description="Stable Diffusion 2.1 - Classic version",
                parameters="860M",
                context_window=None,
                license="CreativeML OpenRAIL-M",
                category="image",
                tags=["classic", "commercial-allowed", "local"]
            ),
        ]
        
        return ProviderConfig(
            name="Stable Diffusion",
            category=ProviderCategory.IMAGE,
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
        Initialize Stable Diffusion provider.
        
        Args:
            base_url: Base URL for local Stable Diffusion API (e.g., ComfyUI)
        """
        if base_url is None:
            base_url = os.getenv("STABLE_DIFFUSION_BASE_URL", self.BASE_URL)
        
        # Create a custom config with the provided base URL
        config = self.get_default_config()
        config.base_url = base_url
        
        super().__init__(config=config)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Stable Diffusion API."""
        headers = super()._get_headers()
        headers["Content-Type"] = "application/json"
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
        Generate an image from a text prompt using Stable Diffusion.
        
        Args:
            prompt: Text description of the image
            model: Model to use (default: stable-diffusion-3.5)
            size: Image size (e.g., "1024x1024")
            quality: Quality setting
            **kwargs: Additional parameters (steps, cfg_scale, seed, etc.)
            
        Returns:
            bytes: Generated image data (PNG)
        """
        # Default model
        if model is None:
            model = "stable-diffusion-3.5"
        
        # Default size
        if size is None:
            size = "1024x1024"
        
        # Parse size
        try:
            width, height = map(int, size.lower().split('x'))
        except:
            width, height = 1024, 1024
        
        # Build payload for ComfyUI API
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": kwargs.get("steps", 20),
            "cfg_scale": kwargs.get("cfg_scale", 7.0),
            "seed": kwargs.get("seed", -1),  # -1 for random
            "model": model,
        }
        
        # Add optional parameters
        if "negative_prompt" in kwargs:
            payload["negative_prompt"] = kwargs["negative_prompt"]
        if "sampler" in kwargs:
            payload["sampler"] = kwargs["sampler"]
        if "scheduler" in kwargs:
            payload["scheduler"] = kwargs["scheduler"]
        
        # ComfyUI API endpoint
        url = f"{self.config.base_url}/sdapi/v1/txt2img"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=300.0  # Image generation can take time
                )
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Unknown error")
                    raise APIError(f"Stable Diffusion API error: {error_msg}")
                
                result = response.json()
                
                # Extract image data
                if "images" in result and len(result["images"]) > 0:
                    # Base64 encoded image data
                    image_data = result["images"][0]
                    return base64.b64decode(image_data)
                
                raise APIError("No image data in response")
                
        except httpx.TimeoutException:
            raise APIError("Image generation timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def generate_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate an image asynchronously.
        
        Args:
            prompt: Text description of the image
            model: Model to use
            size: Image size
            quality: Quality setting
            **kwargs: Additional parameters
            
        Returns:
            str: Job ID for async processing
        """
        # For async generation, we would typically use a different endpoint
        # This is a placeholder implementation
        payload = {
            "prompt": prompt,
            "model": model or "stable-diffusion-3.5",
            "size": size or "1024x1024",
            **kwargs
        }
        
        url = f"{self.config.base_url}/sdapi/v1/txt2img/async"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code != 202:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Unknown error")
                    raise APIError(f"Stable Diffusion async API error: {error_msg}")
                
                result = response.json()
                return result.get("job_id", "")
                
        except httpx.TimeoutException:
            raise APIError("Request timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def get_image(
        self,
        job_id: str
    ) -> bytes:
        """
        Get the generated image from an async job.
        
        Args:
            job_id: Job ID from async generation
            
        Returns:
            bytes: Generated image data
        """
        url = f"{self.config.base_url}/sdapi/v1/txt2img/async/{job_id}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(),
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Unknown error")
                    raise APIError(f"Stable Diffusion get image API error: {error_msg}")
                
                result = response.json()
                
                if "images" in result and len(result["images"]) > 0:
                    image_data = result["images"][0]
                    return base64.b64decode(image_data)
                
                raise APIError("No image data in response")
                
        except httpx.TimeoutException:
            raise APIError("Request timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.session.aclose()


class APIError(Exception):
    """API error exception."""
    pass