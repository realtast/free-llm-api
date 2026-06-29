"""
Base Provider Class

Abstract base class for all API providers with common functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ProviderCategory(Enum):
    """Categories of AI providers."""
    LLM = "llm"
    IMAGE = "image"
    SPEECH = "speech"
    EMBEDDINGS = "embeddings"
    MULTIMODAL = "multimodal"


class ProviderType(Enum):
    """Types of providers."""
    CLOUD = "cloud"
    LOCAL = "local"
    HYBRID = "hybrid"


@dataclass
class RateLimit:
    """Rate limit configuration for a provider."""
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    tokens_per_hour: Optional[int] = None
    tokens_per_day: Optional[int] = None
    neurons_per_day: Optional[int] = None  # For Cloudflare
    burst_limit: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "tokens_per_minute": self.tokens_per_minute,
            "tokens_per_hour": self.tokens_per_hour,
            "tokens_per_day": self.tokens_per_day,
            "neurons_per_day": self.neurons_per_day,
            "burst_limit": self.burst_limit,
        }


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    description: str = ""
    parameters: Optional[str] = None
    context_window: Optional[int] = None
    license: Optional[str] = None
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "context_window": self.context_window,
            "license": self.license,
            "category": self.category,
            "tags": self.tags,
        }


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    name: str
    category: ProviderCategory
    provider_type: ProviderType = ProviderType.CLOUD
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None
    rate_limits: RateLimit = field(default_factory=RateLimit)
    requires_auth: bool = True
    requires_phone_verification: bool = False
    data_training_opt_out_available: bool = False
    commercial_usage_allowed: bool = True
    region: str = "global"
    models: List[ModelInfo] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "provider_type": self.provider_type.value,
            "base_url": self.base_url,
            "api_key_env_var": self.api_key_env_var,
            "rate_limits": self.rate_limits.to_dict(),
            "requires_auth": self.requires_auth,
            "requires_phone_verification": self.requires_phone_verification,
            "data_training_opt_out_available": self.data_training_opt_out_available,
            "commercial_usage_allowed": self.commercial_usage_allowed,
            "region": self.region,
            "models": [model.to_dict() for model in self.models],
        }


class BaseProvider(ABC):
    """Abstract base class for all API providers."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[ProviderConfig] = None):
        """
        Initialize the provider.
        
        Args:
            api_key: API key for authentication
            config: Provider configuration
        """
        self.api_key = api_key
        self.config = config or self.get_default_config()
        self._last_request_time: Optional[float] = None
        self._request_count_today: int = 0
        self._request_count_minute: int = 0
        self._last_minute_reset: float = time.time()
        self._last_day_reset: float = time.time()
        
        # Initialize rate limiting tracking
        self._init_rate_limiting()
        
    @classmethod
    @abstractmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get the default configuration for this provider."""
        pass
    
    def _init_rate_limiting(self):
        """Initialize rate limiting tracking."""
        self._last_request_time = None
        self._request_count_today = 0
        self._request_count_minute = 0
        self._last_minute_reset = time.time()
        self._last_day_reset = time.time()
    
    def _check_rate_limits(self) -> bool:
        """
        Check if rate limits have been exceeded.
        
        Returns:
            bool: True if within rate limits, False otherwise
        """
        current_time = time.time()
        
        # Reset minute counter if needed
        if current_time - self._last_minute_reset >= 60:
            self._request_count_minute = 0
            self._last_minute_reset = current_time
        
        # Reset day counter if needed
        if current_time - self._last_day_reset >= 86400:  # 24 hours
            self._request_count_today = 0
            self._last_day_reset = current_time
        
        # Check requests per minute
        if (self.config.rate_limits.requests_per_minute is not None and 
            self._request_count_minute >= self.config.rate_limits.requests_per_minute):
            logger.warning(f"Rate limit exceeded: {self.config.rate_limits.requests_per_minute} requests/minute")
            return False
        
        # Check requests per day
        if (self.config.rate_limits.requests_per_day is not None and 
            self._request_count_today >= self.config.rate_limits.requests_per_day):
            logger.warning(f"Rate limit exceeded: {self.config.rate_limits.requests_per_day} requests/day")
            return False
        
        return True
    
    def _update_request_counters(self):
        """Update request counters after a successful request."""
        current_time = time.time()
        
        # Reset counters if time periods have passed
        if current_time - self._last_minute_reset >= 60:
            self._request_count_minute = 0
            self._last_minute_reset = current_time
        
        if current_time - self._last_day_reset >= 86400:
            self._request_count_today = 0
            self._last_day_reset = current_time
        
        # Increment counters
        self._request_count_minute += 1
        self._request_count_today += 1
        self._last_request_time = current_time
    
    def _get_headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        if self.api_key and self.config.api_key_env_var:
            # Common header patterns
            if "bearer" in self.config.api_key_env_var.lower():
                headers["Authorization"] = f"Bearer {self.api_key}"
            elif "api-key" in self.config.api_key_env_var.lower():
                headers["X-API-Key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    @abstractmethod
    async def chat(
        self, 
        model: str, 
        messages: Union[str, List[Dict[str, Any]]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send a chat completion request.
        
        Args:
            model: Model name or ID
            messages: Chat messages (string or list of message dicts)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            str: Generated text response
        """
        pass
    
    @abstractmethod
    async def complete(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send a text completion request.
        
        Args:
            model: Model name or ID
            prompt: Text prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            str: Generated text
        """
        pass
    
    def get_config(self) -> ProviderConfig:
        """Get the provider configuration."""
        return self.config
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return [model.name for model in self.config.models]
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        for model in self.config.models:
            if model.name == model_name:
                return model
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', category='{self.config.category.value}')"


class BaseLLMProvider(BaseProvider):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def chat(
        self, 
        model: str, 
        messages: Union[str, List[Dict[str, Any]]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        pass
    
    @abstractmethod
    async def complete(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        pass
    
    async def embed(
        self,
        model: str,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        Generate embeddings for text (if supported).
        
        Args:
            model: Embedding model name
            text: Text to embed
            **kwargs: Additional parameters
            
        Returns:
            List[float]: Embedding vector
        """
        raise NotImplementedError("Embeddings not supported by this provider")


class BaseImageProvider(BaseProvider):
    """Base class for image generation providers."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image
            model: Model to use
            size: Image size (e.g., "1024x1024")
            quality: Quality setting
            **kwargs: Additional parameters
            
        Returns:
            bytes: Generated image data
        """
        pass


class BaseSpeechProvider(BaseProvider):
    """Base class for speech providers."""
    
    @abstractmethod
    async def transcribe(
        self,
        audio_data: Union[str, bytes],
        model: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Path to audio file or audio data bytes
            model: Model to use
            language: Language hint
            **kwargs: Additional parameters
            
        Returns:
            str: Transcribed text
        """
        pass
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            model: Model to use
            voice: Voice to use
            **kwargs: Additional parameters
            
        Returns:
            bytes: Audio data
        """
        pass


class BaseEmbeddingProvider(BaseProvider):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def embed(
        self,
        model: str,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            model: Embedding model name
            text: Text to embed
            **kwargs: Additional parameters
            
        Returns:
            List[float]: Embedding vector
        """
        pass
    
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
        # Default implementation: call embed for each text
        return [await self.embed(model, text, **kwargs) for text in texts]