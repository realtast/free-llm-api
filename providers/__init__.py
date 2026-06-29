"""
Free LLM API Providers Package

This package contains implementations for various free LLM API providers
organized by category: LLM, Image, Speech, and Embeddings.
"""

from .llm import (
    GoogleAIStudioProvider,
    GroqProvider,
    MistralProvider,
    OpenRouterProvider,
    CerebrasProvider,
    CohereProvider,
    CloudflareProvider,
    HuggingFaceProvider,
    NVIDIAProvider,
    VercelProvider,
    GitHubModelsProvider,
)
from .image import (
    StableDiffusionProvider,
    FLUXProvider,
    GoogleGeminiImageProvider,
    AdobeFireflyProvider,
    MidjourneyProvider,
    GPTImageProvider,
    ImagenProvider,
)
from .speech import (
    GroqSpeechProvider,
    GoogleTTSProvider,
    WhisperLocalProvider,
)
from .embeddings import (
    HuggingFaceEmbeddingsProvider,
    CloudflareEmbeddingsProvider,
    OpenRouterEmbeddingsProvider,
)

__all__ = [
    # LLM Providers
    "GoogleAIStudioProvider",
    "GroqProvider",
    "MistralProvider",
    "OpenRouterProvider",
    "CerebrasProvider",
    "CohereProvider",
    "CloudflareProvider",
    "HuggingFaceProvider",
    "NVIDIAProvider",
    "VercelProvider",
    "GitHubModelsProvider",
    # Image Providers
    "StableDiffusionProvider",
    "FLUXProvider",
    "GoogleGeminiImageProvider",
    "AdobeFireflyProvider",
    "MidjourneyProvider",
    "GPTImageProvider",
    "ImagenProvider",
    # Speech Providers
    "GroqSpeechProvider",
    "GoogleTTSProvider",
    "WhisperLocalProvider",
    # Embedding Providers
    "HuggingFaceEmbeddingsProvider",
    "CloudflareEmbeddingsProvider",
    "OpenRouterEmbeddingsProvider",
]