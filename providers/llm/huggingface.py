"""
Hugging Face Provider

Implementation for Hugging Face Inference API.
Serverless inference for open models with $0.10/month credits.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseLLMProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """
    Hugging Face Inference API Provider.
    
    Features:
    - Access to latest open models
    - Hugging Face ecosystem integration
    - Serverless inference
    - Models smaller than 10GB supported
    
    Limitations:
    - Very limited credits ($0.10/month)
    - Not suitable for production
    
    Best for: Testing, model evaluation
    """
    
    BASE_URL = "https://api-inference.huggingface.co"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Hugging Face."""
        # Note: Hugging Face has many models, we'll list some popular ones
        models = [
            ModelInfo(
                name="meta-llama/Llama-3.1-8B-Instruct",
                description="Llama 3.1 8B Instruct",
                parameters="8B",
                context_window=128000,
                license="Llama 3.1",
                category="general",
                tags=["llama", "meta"]
            ),
            ModelInfo(
                name="meta-llama/Llama-3.2-11B-Vision-Instruct",
                description="Llama 3.2 11B Vision Instruct",
                parameters="11B",
                context_window=128000,
                license="Llama 3.2",
                category="vision",
                tags=["llama", "meta", "vision"]
            ),
            ModelInfo(
                name="mistralai/Mistral-7B-Instruct-v0.3",
                description="Mistral 7B Instruct v0.3",
                parameters="7B",
                context_window=32768,
                license="Apache 2.0",
                category="general",
                tags=["mistral", "apache-2.0"]
            ),
            ModelInfo(
                name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                description="Mixtral 8x7B Instruct v0.1",
                parameters="47B",
                context_window=32768,
                license="Apache 2.0",
                category="general",
                tags=["mistral", "mixtral", "apache-2.0"]
            ),
            ModelInfo(
                name="Qwen/Qwen2.5-7B-Instruct",
                description="Qwen 2.5 7B Instruct",
                parameters="7B",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["qwen", "apache-2.0"]
            ),
            ModelInfo(
                name="Qwen/Qwen2.5-Coder-7B-Instruct",
                description="Qwen 2.5 Coder 7B Instruct",
                parameters="7B",
                context_window=128000,
                license="Apache 2.0",
                category="coding",
                tags=["qwen", "coding", "apache-2.0"]
            ),
            ModelInfo(
                name="google/gemma-2-9b-it",
                description="Gemma 2 9B IT",
                parameters="9B",
                context_window=8192,
                license="Apache 2.0",
                category="general",
                tags=["google", "gemma", "apache-2.0"]
            ),
            ModelInfo(
                name="google/gemma-2-27b-it",
                description="Gemma 2 27B IT",
                parameters="27B",
                context_window=8192,
                license="Apache 2.0",
                category="general",
                tags=["google", "gemma", "apache-2.0"]
            ),
            ModelInfo(
                name="phi-3-medium-128k-instruct",
                description="Phi 3 Medium 128K Instruct",
                parameters="14B",
                context_window=128000,
                license="MIT",
                category="general",
                tags=["phi", "microsoft"]
            ),
            ModelInfo(
                name="phi-3-mini-128k-instruct",
                description="Phi 3 Mini 128K Instruct",
                parameters="3.8B",
                context_window=128000,
                license="MIT",
                category="general",
                tags=["phi", "microsoft", "small"]
            ),
        ]
        
        return ProviderConfig(
            name="Hugging Face",
            category=ProviderCategory.LLM,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="HUGGINGFACE_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=10,  # Estimated
                requests_per_day=10,  # Very limited due to $0.10 credits
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
        Initialize Hugging Face provider.
        
        Args:
            api_key: Hugging Face API key
        """
        if api_key is None:
            api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Hugging Face API."""
        headers = super()._get_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def chat(
        self,
        model: str,
        messages: Union[str, List[Dict[str, Any]]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send a chat completion request to Hugging Face.
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("Hugging Face rate limit exceeded")
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Hugging Face uses a chat completion format
        hf_messages = []
        for msg in messages:
            if isinstance(msg, str):
                hf_messages.append({"role": "user", "content": msg})
            else:
                hf_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        payload = {
            "inputs": hf_messages,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens or 512,
            }
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            payload["parameters"]["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["parameters"]["top_k"] = kwargs["top_k"]
        if "seed" in kwargs:
            payload["parameters"]["seed"] = kwargs["seed"]
        if "return_full_text" in kwargs:
            payload["parameters"]["return_full_text"] = kwargs["return_full_text"]
        
        # Clean model name for URL
        clean_model = model.replace("/", "--")
        url = f"{self.BASE_URL}/models/{clean_model}"
        
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
                    raise APIError(f"Hugging Face API error: {error_msg}")
                
                result = response.json()
                
                # Extract text from response
                if "generated_text" in result:
                    return result["generated_text"]
                elif "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0].get("message", {}).get("content", "")
                
                return ""
                
        except httpx.TimeoutException:
            raise APIError("Request timeout")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def complete(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Send a text completion request."""
        # For Hugging Face, we can use the same endpoint for completions
        return await self.chat(model, prompt, temperature, max_tokens, **kwargs)
    
    async def embed(
        self,
        model: str,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        Generate embeddings using Hugging Face models.
        """
        # Use the embeddings endpoint
        clean_model = model.replace("/", "--")
        url = f"{self.BASE_URL}/models/{clean_model}"
        
        payload = {
            "inputs": text,
            "parameters": kwargs
        }
        
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
                    raise APIError(f"Hugging Face embed API error: {error_msg}")
                
                result = response.json()
                
                # Extract embedding from response
                if "embedding" in result:
                    return result["embedding"]
                elif "outputs" in result and len(result["outputs"]) > 0:
                    return result["outputs"][0]
                
                return []
                
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


class RateLimitExceededError(Exception):
    """Rate limit exceeded exception."""
    pass