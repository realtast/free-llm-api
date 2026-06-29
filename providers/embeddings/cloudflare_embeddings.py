"""
Cloudflare Embeddings Provider

Implementation for Cloudflare Workers AI Embeddings.
Edge-based embeddings with 10,000 neurons/day limit.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseEmbeddingProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class CloudflareEmbeddingsProvider(BaseEmbeddingProvider):
    """
    Cloudflare Workers AI Embeddings Provider.
    
    Features:
    - Edge deployment
    - Serverless
    - Various embedding models available
    - Shared neuron limit with other Cloudflare AI models
    
    Limitations:
    - Neuron-based limits can be confusing
    - Shared quota with other models
    
    Best for: Serverless applications
    """
    
    BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Cloudflare Embeddings."""
        models = [
            ModelInfo(
                name="@cf/baai/bge-small-en-v1.5",
                description="BGE Small EN v1.5 - English embedding",
                parameters="110M",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["english", "bge", "mit", "edge"]
            ),
            ModelInfo(
                name="@cf/baai/bge-base-en-v1.5",
                description="BGE Base EN v1.5 - English embedding",
                parameters="420M",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["english", "bge", "mit", "edge"]
            ),
            ModelInfo(
                name="@cf/baai/bge-large-en-v1.5",
                description="BGE Large EN v1.5 - English embedding",
                parameters="1.5B",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["english", "bge", "mit", "edge", "large"]
            ),
            ModelInfo(
                name="@cf/sentence-transformers/all-mpnet-base-v2",
                description="All MPNet Base v2 - General-purpose embedding",
                parameters="420M",
                context_window=None,
                license="Apache 2.0",
                category="embeddings",
                tags=["general-purpose", "apache-2.0", "edge"]
            ),
            ModelInfo(
                name="@cf/sentence-transformers/all-MiniLM-L6-v2",
                description="All MiniLM L6 v2 - Lightweight embedding",
                parameters="80M",
                context_window=None,
                license="Apache 2.0",
                category="embeddings",
                tags=["lightweight", "fast", "apache-2.0", "edge"]
            ),
            ModelInfo(
                name="@cf/intfloat/e5-small-v2",
                description="E5 Small v2 - Multilingual embedding",
                parameters="100M",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["multilingual", "e5", "mit", "edge"]
            ),
            ModelInfo(
                name="@cf/intfloat/e5-base-v2",
                description="E5 Base v2 - Multilingual embedding",
                parameters="420M",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["multilingual", "e5", "mit", "edge"]
            ),
            ModelInfo(
                name="@cf/intfloat/e5-large-v2",
                description="E5 Large v2 - Multilingual embedding",
                parameters="1.5B",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["multilingual", "e5", "mit", "edge", "large"]
            ),
        ]
        
        return ProviderConfig(
            name="Cloudflare Embeddings",
            category=ProviderCategory.EMBEDDINGS,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="CLOUDFLARE_API_KEY",
            rate_limits=RateLimit(
                neurons_per_day=10000,  # Shared with other Cloudflare AI models
            ),
            requires_auth=True,
            requires_phone_verification=False,
            data_training_opt_out_available=True,
            commercial_usage_allowed=True,
            region="global",
            models=models
        )
    
    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None):
        """
        Initialize Cloudflare Embeddings provider.
        
        Args:
            api_key: Cloudflare API key
            account_id: Cloudflare account ID
        """
        if api_key is None:
            api_key = os.getenv("CLOUDFLARE_API_KEY")
        if account_id is None:
            account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        
        super().__init__(api_key=api_key)
        self.account_id = account_id
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Cloudflare API."""
        headers = super()._get_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _get_model_url(self, model: str) -> str:
        """Get the URL for a specific embedding model."""
        # Clean model name for URL
        clean_model = model.replace("@cf/", "").replace("/", "-")
        return f"{self.BASE_URL}/{self.account_id}/ai/v1/{clean_model}"
    
    async def embed(
        self,
        model: str,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        Generate embeddings for text using Cloudflare models.
        
        Args:
            model: Embedding model name
            text: Text to embed
            **kwargs: Additional parameters
            
        Returns:
            List[float]: Embedding vector
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("Cloudflare Embeddings rate limit exceeded")
        
        payload = {
            "text": text,
        }
        
        # Add optional parameters
        if "normalize" in kwargs:
            payload["normalize"] = kwargs["normalize"]
        
        url = self._get_model_url(model)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=60.0
                )
                
                self._update_request_counters()
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Unknown error")
                    raise APIError(f"Cloudflare Embeddings API error: {error_msg}")
                
                result = response.json()
                
                # Extract embedding from response
                if "embedding" in result:
                    return result["embedding"]
                elif "result" in result:
                    if "embedding" in result["result"]:
                        return result["result"]["embedding"]
                    elif "output" in result["result"]:
                        return result["result"]["output"]
                
                raise APIError("No embedding data in response")
                
        except httpx.TimeoutException:
            raise APIError("Embedding generation timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def embed_batch(
        self,
        model: str,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            model: Embedding model name
            texts: List of texts to embed
            **kwargs: Additional parameters
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("Cloudflare Embeddings rate limit exceeded")
        
        payload = {
            "texts": texts,
        }
        
        # Add optional parameters
        if "normalize" in kwargs:
            payload["normalize"] = kwargs["normalize"]
        
        url = self._get_model_url(model)
        
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
                    error_msg = error_data.get("error", "Unknown error")
                    raise APIError(f"Cloudflare Embeddings batch API error: {error_msg}")
                
                result = response.json()
                
                # Extract embeddings from response
                if "embeddings" in result:
                    return result["embeddings"]
                elif "result" in result:
                    if "embeddings" in result["result"]:
                        return result["result"]["embeddings"]
                    elif "outputs" in result["result"]:
                        return result["result"]["outputs"]
                
                raise APIError("No embedding data in response")
                
        except httpx.TimeoutException:
            raise APIError("Batch embedding generation timeout")
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