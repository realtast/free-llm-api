"""
FLUX Provider

Implementation for FLUX image generation.
Highest technical quality in 2026, free for local use.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import base64

from ..base_provider import BaseImageProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class FLUXProvider(BaseImageProvider):
    """
    FLUX Image Generation Provider.
    
    Features:
    - Highest technical quality in 2026
    - Free for local use
    - 4.5-second generation time
    - Best for realism
    - Commercial use allowed
    
    Limitations:
    - Requires capable hardware
    - Local deployment required
    
    Best for: High-quality image generation, commercial projects
    """
    
    BASE_URL = "http://localhost:7860"  # Default for local ComfyUI with FLUX
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for FLUX."""
        models = [
            ModelInfo(
                name="flux.1.1-pro",
                description="FLUX.1.1 Pro - Highest technical quality",
                parameters="12B",
                context_window=None,
                license="Apache 2.0",
                category="image",
                tags=["top-pick", "highest-quality", "realism", "commercial-allowed", "local"]
            ),
            ModelInfo(
                name="flux.1.1",
                description="FLUX.1.1 - High quality",
                parameters="12B",
                context_window=None,
                license="Apache 2.0",
                category="image",
                tags=["high-quality", "realism", "commercial-allowed", "local"]
            ),
            ModelInfo(
                name="flux.1-dev",
                description="FLUX.1 Dev - Development version",
                parameters="12B",
                context_window=None,
                license="Apache 2.0",
                category="image",
                tags=["development", "realism", "commercial-allowed", "local"]
            ),
        ]
        
        return ProviderConfig(
            name="FLUX",
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
        Initialize FLUX provider.
        
        Args:
            base_url: Base URL for local FLUX API (e.g., ComfyUI)
        """
        if base_url is None:
            base_url = os.getenv("FLUX_BASE_URL", self.BASE_URL)
        
        # Create a custom config with the provided base URL
        config = self.get_default_config()
        config.base_url = base_url
        
        super().__init__(config=config)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for FLUX API."""
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
        Generate an image from a text prompt using FLUX.
        
        Args:
            prompt: Text description of the image
            model: Model to use (default: flux.1.1-pro)
            size: Image size (e.g., "1024x1024")
            quality: Quality setting
            **kwargs: Additional parameters (steps, cfg_scale, seed, etc.)
            
        Returns:
            bytes: Generated image data (PNG)
        """
        # Default model
        if model is None:
            model = "flux.1.1-pro"
        
        # Default size
        if size is None:
            size = "1024x1024"
        
        # Parse size
        try:
            width, height = map(int, size.lower().split('x'))
        except:
            width, height = 1024, 1024
        
        # FLUX typically uses different parameters than Stable Diffusion
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "model": model,
            "steps": kwargs.get("steps", 28),  # FLUX often uses 28-50 steps
            "cfg_scale": kwargs.get("cfg_scale", 7.0),
            "seed": kwargs.get("seed", -1),  # -1 for random
        }
        
        # Add optional parameters
        if "negative_prompt" in kwargs:
            payload["negative_prompt"] = kwargs["negative_prompt"]
        if "sampler" in kwargs:
            payload["sampler"] = kwargs["sampler"]
        if "scheduler" in kwargs:
            payload["scheduler"] = kwargs["scheduler"]
        
        # FLUX through ComfyUI API
        url = f"{self.config.base_url}/sdapi/v1/txt2img"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=300.0  # FLUX generation can take time
                )
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Unknown error")
                    raise APIError(f"FLUX API error: {error_msg}")
                
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
    
    async def close(self):
        """Close the HTTP client."""
        await self.session.aclose()


class APIError(Exception):
    """API error exception."""
    pass