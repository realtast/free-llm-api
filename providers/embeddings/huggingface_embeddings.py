"""
Hugging Face Embeddings Provider

Implementation for Hugging Face Embeddings.
Most popular open-source embeddings with free local use.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseEmbeddingProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingsProvider(BaseEmbeddingProvider):
    """
    Hugging Face Embeddings Provider.
    
    Features:
    - Huge model selection
    - Open-source
    - Free for local use
    - Custom embedding solutions
    
    Limitations:
    - Requires local deployment
    
    Best for: Custom embedding solutions
    """
    
    BASE_URL = "http://localhost:8000"  # Default for local embedding API
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Hugging Face Embeddings."""
        models = [
            ModelInfo(
                name="sentence-transformers/all-mpnet-base-v2",
                description="All MPNet Base v2 - Popular general-purpose embedding",
                parameters="420M",
                context_window=None,
                license="Apache 2.0",
                category="embeddings",
                tags=["top-pick", "general-purpose", "apache-2.0"]
            ),
            ModelInfo(
                name="sentence-transformers/all-MiniLM-L6-v2",
                description="All MiniLM L6 v2 - Lightweight embedding",
                parameters="80M",
                context_window=None,
                license="Apache 2.0",
                category="embeddings",
                tags=["lightweight", "fast", "apache-2.0"]
            ),
            ModelInfo(
                name="BAAI/bge-small-en-v1.5",
                description="BGE Small EN v1.5 - English embedding",
                parameters="110M",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["english", "bge", "mit"]
            ),
            ModelInfo(
                name="BAAI/bge-base-en-v1.5",
                description="BGE Base EN v1.5 - English embedding",
                parameters="420M",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["english", "bge", "mit"]
            ),
            ModelInfo(
                name="BAAI/bge-large-en-v1.5",
                description="BGE Large EN v1.5 - English embedding",
                parameters="1.5B",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["english", "bge", "mit", "large"]
            ),
            ModelInfo(
                name="intfloat/e5-small-v2",
                description="E5 Small v2 - Multilingual embedding",
                parameters="100M",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["multilingual", "e5", "mit"]
            ),
            ModelInfo(
                name="intfloat/e5-base-v2",
                description="E5 Base v2 - Multilingual embedding",
                parameters="420M",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["multilingual", "e5", "mit"]
            ),
            ModelInfo(
                name="intfloat/e5-large-v2",
                description="E5 Large v2 - Multilingual embedding",
                parameters="1.5B",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["multilingual", "e5", "mit", "large"]
            ),
            ModelInfo(
                name="thenlper/gte-small",
                description="GTE Small - General text embedding",
                parameters="100M",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["general", "gte", "mit"]
            ),
            ModelInfo(
                name="thenlper/gte-base",
                description="GTE Base - General text embedding",
                parameters="420M",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["general", "gte", "mit"]
            ),
            ModelInfo(
                name="thenlper/gte-large",
                description="GTE Large - General text embedding",
                parameters="1.5B",
                context_window=None,
                license="MIT",
                category="embeddings",
                tags=["general", "gte", "mit", "large"]
            ),
        ]
        
        return ProviderConfig(
            name="Hugging Face Embeddings",
            category=ProviderCategory.EMBEDDINGS,
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
        Initialize Hugging Face Embeddings provider.
        
        Args:
            base_url: Base URL for local embedding API
        """
        if base_url is None:
            base_url = os.getenv("HUGGINGFACE_EMBEDDINGS_BASE_URL", self.BASE_URL)
        
        # Create a custom config with the provided base URL
        config = self.get_default_config()
        config.base_url = base_url
        
        super().__init__(config=config)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Hugging Face Embeddings API."""
        headers = super()._get_headers()
        headers["Content-Type"] = "application/json"
        return headers
    
    async def embed(
        self,
        model: str,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        Generate embeddings for text using Hugging Face models.
        
        Args:
            model: Embedding model name
            text: Text to embed
            **kwargs: Additional parameters
            
        Returns:
            List[float]: Embedding vector
        """
        # Clean model name for URL
        clean_model = model.replace("/", "--")
        
        payload = {
            "inputs": text,
            "model": model,
        }
        
        # Add optional parameters
        if "normalize" in kwargs:
            payload["normalize"] = kwargs["normalize"]
        if "batch_size" in kwargs:
            payload["batch_size"] = kwargs["batch_size"]
        
        url = f"{self.config.base_url}/embed"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Unknown error")
                    raise APIError(f"Hugging Face Embeddings API error: {error_msg}")
                
                result = response.json()
                
                # Extract embedding from response
                if "embedding" in result:
                    return result["embedding"]
                elif "outputs" in result and len(result["outputs"]) > 0:
                    return result["outputs"][0]
                
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
        # Clean model name for URL
        clean_model = model.replace("/", "--")
        
        payload = {
            "inputs": texts,
            "model": model,
        }
        
        # Add optional parameters
        if "normalize" in kwargs:
            payload["normalize"] = kwargs["normalize"]
        if "batch_size" in kwargs:
            payload["batch_size"] = kwargs["batch_size"]
        
        url = f"{self.config.base_url}/embed_batch"
        
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
                    error_msg = error_data.get("error", "Unknown error")
                    raise APIError(f"Hugging Face Embeddings batch API error: {error_msg}")
                
                result = response.json()
                
                # Extract embeddings from response
                if "embeddings" in result:
                    return result["embeddings"]
                elif "outputs" in result:
                    return result["outputs"]
                
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