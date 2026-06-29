"""
Cloudflare Workers AI Provider

Implementation for Cloudflare Workers AI.
Serverless AI at the edge with unique neuron-based pricing: 10,000 neurons/day.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseLLMProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class CloudflareProvider(BaseLLMProvider):
    """
    Cloudflare Workers AI Provider.
    
    Features:
    - Edge deployment
    - Serverless architecture
    - Unique neuron-based pricing model
    - 30+ models available
    
    Limitations:
    - Neuron-based limits can be confusing
    - Different pricing model than traditional APIs
    
    Best for: Edge computing, serverless applications
    """
    
    BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Cloudflare Workers AI."""
        models = [
            ModelInfo(
                name="@cf/aisingapore/gemma-sea-lion-v4-27b-it",
                description="Gemma Sea Lion V4 27B IT",
                parameters="27B",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["gemma", "sea-lion", "apache-2.0"]
            ),
            ModelInfo(
                name="@cf/google/gemma-4-26b-a4b-it",
                description="Google Gemma 4 26B A4B IT",
                parameters="26B",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["google", "gemma", "apache-2.0"]
            ),
            ModelInfo(
                name="@cf/google/gemma-4-31b-it",
                description="Google Gemma 4 31B IT",
                parameters="31B",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["google", "gemma", "apache-2.0"]
            ),
            ModelInfo(
                name="@cf/ibm-granite/granite-4.0-h-micro",
                description="IBM Granite 4.0 H Micro",
                parameters="Unknown",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["ibm", "granite", "apache-2.0"]
            ),
            ModelInfo(
                name="@cf/moonshotai/kimi-k2.6",
                description="Moonshot AI Kimi K2.6",
                parameters="Unknown",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["moonshot", "kimi", "apache-2.0"]
            ),
            ModelInfo(
                name="@cf/moonshotai/kimi-k2.7-code",
                description="Moonshot AI Kimi K2.7 Code",
                parameters="Unknown",
                context_window=128000,
                license="Apache 2.0",
                category="coding",
                tags=["moonshot", "kimi", "coding", "apache-2.0"]
            ),
            ModelInfo(
                name="@cf/nvidia/nemotron-3-120b-a12b",
                description="NVIDIA Nemotron 3 120B A12B",
                parameters="120B",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["nvidia", "nemotron"]
            ),
            ModelInfo(
                name="@cf/openai/gpt-oss-120b",
                description="OpenAI GPT-OSS 120B",
                parameters="120B",
                context_window=128000,
                license="MIT",
                category="general",
                tags=["openai", "open-model"]
            ),
            ModelInfo(
                name="@cf/openai/gpt-oss-20b",
                description="OpenAI GPT-OSS 20B",
                parameters="20B",
                context_window=128000,
                license="MIT",
                category="general",
                tags=["openai", "open-model"]
            ),
            ModelInfo(
                name="@cf/qwen/qwen3-30b-a3b-fp8",
                description="Qwen 3 30B A3B FP8",
                parameters="30B",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["qwen", "apache-2.0"]
            ),
            ModelInfo(
                name="@cf/qwen/glm-4.7-flash",
                description="Qwen GLM 4.7 Flash",
                parameters="Unknown",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["qwen", "glm", "apache-2.0"]
            ),
            ModelInfo(
                name="@cf/qwen/glm-5.2",
                description="Qwen GLM 5.2",
                parameters="Unknown",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["qwen", "glm", "apache-2.0"]
            ),
            ModelInfo(
                name="llama-2-7b",
                description="Llama 2 7B",
                parameters="7B",
                context_window=4096,
                license="Llama 2",
                category="general",
                tags=["llama"]
            ),
            ModelInfo(
                name="llama-3.1-8b",
                description="Llama 3.1 8B",
                parameters="8B",
                context_window=128000,
                license="Llama 3.1",
                category="general",
                tags=["llama"]
            ),
            ModelInfo(
                name="llama-3.2-11b-vision",
                description="Llama 3.2 11B Vision",
                parameters="11B",
                context_window=128000,
                license="Llama 3.2",
                category="vision",
                tags=["llama", "vision"]
            ),
            ModelInfo(
                name="llama-3.3-70b",
                description="Llama 3.3 70B",
                parameters="70B",
                context_window=128000,
                license="Llama 3.3",
                category="general",
                tags=["llama"]
            ),
            ModelInfo(
                name="llama-4-scout",
                description="Llama 4 Scout",
                parameters="17B",
                context_window=10000000,
                license="Llama 4",
                category="general",
                tags=["llama", "long-context"]
            ),
            ModelInfo(
                name="mistral-7b",
                description="Mistral 7B",
                parameters="7B",
                context_window=32768,
                license="Apache 2.0",
                category="general",
                tags=["mistral", "apache-2.0"]
            ),
            ModelInfo(
                name="mistral-small-3.1-24b",
                description="Mistral Small 3.1 24B",
                parameters="24B",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["mistral", "apache-2.0"]
            ),
            ModelInfo(
                name="qwen-2.5-coder-32b",
                description="Qwen 2.5 Coder 32B",
                parameters="32B",
                context_window=128000,
                license="Apache 2.0",
                category="coding",
                tags=["qwen", "coding", "apache-2.0"]
            ),
            ModelInfo(
                name="qwen-qwq-32b",
                description="Qwen QwQ 32B",
                parameters="32B",
                context_window=128000,
                license="Apache 2.0",
                category="general",
                tags=["qwen", "apache-2.0"]
            ),
        ]
        
        return ProviderConfig(
            name="Cloudflare Workers AI",
            category=ProviderCategory.LLM,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="CLOUDFLARE_API_KEY",
            rate_limits=RateLimit(
                neurons_per_day=10000,
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
        Initialize Cloudflare Workers AI provider.
        
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
        """Get the URL for a specific model."""
        # Clean model name for URL
        clean_model = model.replace("@cf/", "").replace("/", "-")
        return f"{self.BASE_URL}/{self.account_id}/ai/v1/{clean_model}"
    
    async def chat(
        self,
        model: str,
        messages: Union[str, List[Dict[str, Any]]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send a chat completion request to Cloudflare Workers AI.
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("Cloudflare Workers AI rate limit exceeded")
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Cloudflare uses a different message format
        cf_messages = []
        for msg in messages:
            if isinstance(msg, str):
                cf_messages.append({"role": "user", "content": msg})
            else:
                cf_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        payload = {
            "messages": cf_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]
        
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
                    raise APIError(f"Cloudflare API error: {error_msg}")
                
                result = response.json()
                
                # Extract text from response
                if "result" in result:
                    if "response" in result["result"]:
                        return result["result"]["response"]
                    elif "choices" in result["result"] and len(result["result"]["choices"]) > 0:
                        return result["result"]["choices"][0].get("message", {}).get("content", "")
                
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
        return await self.chat(model, prompt, temperature, max_tokens, **kwargs)
    
    async def close(self):
        """Close the HTTP client."""
        await self.session.aclose()


class APIError(Exception):
    """API error exception."""
    pass


class RateLimitExceededError(Exception):
    """Rate limit exceeded exception."""
    pass