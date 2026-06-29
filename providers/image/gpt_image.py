"""
GPT Image Provider

Implementation for OpenAI GPT Image Generation.
Best for prompt execution, successor to DALL-E 3.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import base64

from ..base_provider import BaseImageProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class GPTImageProvider(BaseImageProvider):
    """
    OpenAI GPT Image Generation Provider.
    
    Features:
    - Best for prompt execution
    - Understands complex instructions best
    - Successor to GPT Image 1.5 and DALL-E 3
    
    Limitations:
    - API costs may apply (free tier unclear)
    - May require OpenAI account
    
    Best for: Complex instructions, precise requirements
    """
    
    BASE_URL = "https://api.openai.com/v1"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for GPT Image."""
        models = [
            ModelInfo(
                name="gpt-image-2",
                description="GPT Image 2 - Best for prompt execution",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["top-pick", "openai", "prompt-execution", "complex-instructions"]
            ),
            ModelInfo(
                name="gpt-image-1.5",
                description="GPT Image 1.5 - Previous version",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["openai", "prompt-execution"]
            ),
            ModelInfo(
                name="dall-e-3",
                description="DALL-E 3 - Legacy (deprecated May 12, 2026)",
                parameters="Unknown",
                context_window=None,
                license="Proprietary",
                category="image",
                tags=["openai", "legacy", "deprecated"]
            ),
        ]
        
        return ProviderConfig(
            name="GPT Image",
            category=ProviderCategory.IMAGE,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="OPENAI_API_KEY",
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
        Initialize GPT Image provider.
        
        Args:
            api_key: OpenAI API key
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI API."""
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
        Generate an image from a text prompt using GPT Image.
        
        Args:
            prompt: Text description of the image
            model: Model to use (default: gpt-image-2)
            size: Image size (e.g., "1024x1024")
            quality: Quality setting
            **kwargs: Additional parameters
            
        Returns:
            bytes: Generated image data (PNG)
        """
        # Default model
        if model is None:
            model = "gpt-image-2"
        
        # Default size
        if size is None:
            size = "1024x1024"
        
        # Parse size
        try:
            width, height = map(int, size.lower().split('x'))
        except:
            width, height = 1024, 1024
        
        # OpenAI GPT Image payload
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
        }
        
        # Add optional parameters
        if "quality" in kwargs:
            payload["quality"] = kwargs["quality"]
        if "n" in kwargs:
            payload["n"] = kwargs["n"]  # Number of images
        if "style" in kwargs:
            payload["style"] = kwargs["style"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]
        
        url = f"{self.BASE_URL}/images/generations"
        
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
                    raise APIError(f"GPT Image API error: {error_msg}")
                
                result = response.json()
                
                # Extract image data
                if "data" in result and len(result["data"]) > 0:
                    image_data = result["data"][0].get("b64_json", "")
                    if image_data:
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