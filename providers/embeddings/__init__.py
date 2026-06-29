"""
Embedding Providers Package

Collection of free embedding API providers.
"""

from .huggingface_embeddings import HuggingFaceEmbeddingsProvider
from .cloudflare_embeddings import CloudflareEmbeddingsProvider
from .openrouter_embeddings import OpenRouterEmbeddingsProvider

__all__ = [
    "HuggingFaceEmbeddingsProvider",
    "CloudflareEmbeddingsProvider",
    "OpenRouterEmbeddingsProvider",
]