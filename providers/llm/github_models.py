"""
GitHub Models Provider

Implementation for GitHub Models via Copilot.
Access through GitHub Copilot with 40+ models available.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx

from ..base_provider import BaseLLMProvider, ProviderConfig, ProviderCategory, ProviderType, ModelInfo, RateLimit

logger = logging.getLogger(__name__)


class GitHubModelsProvider(BaseLLMProvider):
    """
    GitHub Models Provider via Copilot.
    
    Features:
    - Huge model selection (40+ models)
    - GitHub integration
    - Access to latest models from various providers
    
    Limitations:
    - Requires Copilot subscription (not fully free)
    - Dependent on Copilot subscription tier
    
    Best for: GitHub users, development workflows
    """
    
    BASE_URL = "https://api.github.com/copilot/v1"
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for GitHub Models."""
        models = [
            ModelInfo(
                name="codestral-25.01",
                description="Codestral 25.01",
                parameters="25B",
                context_window=32768,
                license="Apache 2.0",
                category="coding",
                tags=["codestral", "mistral", "coding", "apache-2.0"]
            ),
            ModelInfo(
                name="cohere-command-a",
                description="Cohere Command A",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["cohere", "command"]
            ),
            ModelInfo(
                name="deepseek-r1",
                description="DeepSeek R1",
                parameters="Unknown",
                context_window=1000000,
                license="MIT",
                category="reasoning",
                tags=["deepseek", "reasoning"]
            ),
            ModelInfo(
                name="deepseek-r1-0528",
                description="DeepSeek R1 0528",
                parameters="Unknown",
                context_window=1000000,
                license="MIT",
                category="reasoning",
                tags=["deepseek", "reasoning"]
            ),
            ModelInfo(
                name="deepseek-v3-0324",
                description="DeepSeek V3 0324",
                parameters="Unknown",
                context_window=128000,
                license="MIT",
                category="general",
                tags=["deepseek"]
            ),
            ModelInfo(
                name="llama-4-maverick-17b",
                description="Llama 4 Maverick 17B",
                parameters="17B",
                context_window=10000000,
                license="Llama 4",
                category="general",
                tags=["llama", "meta", "long-context"]
            ),
            ModelInfo(
                name="llama-4-scout-17b",
                description="Llama 4 Scout 17B",
                parameters="17B",
                context_window=10000000,
                license="Llama 4",
                category="general",
                tags=["llama", "meta", "long-context"]
            ),
            ModelInfo(
                name="llama-3.2-11b-vision",
                description="Llama 3.2 11B Vision",
                parameters="11B",
                context_window=128000,
                license="Llama 3.2",
                category="vision",
                tags=["llama", "meta", "vision"]
            ),
            ModelInfo(
                name="llama-3.2-90b-vision",
                description="Llama 3.2 90B Vision",
                parameters="90B",
                context_window=128000,
                license="Llama 3.2",
                category="vision",
                tags=["llama", "meta", "vision"]
            ),
            ModelInfo(
                name="llama-3.3-70b",
                description="Llama 3.3 70B",
                parameters="70B",
                context_window=128000,
                license="Llama 3.3",
                category="general",
                tags=["llama", "meta"]
            ),
            ModelInfo(
                name="meta-llama-3.1-405b",
                description="Meta Llama 3.1 405B",
                parameters="405B",
                context_window=128000,
                license="Llama 3.1",
                category="general",
                tags=["llama", "meta", "large"]
            ),
            ModelInfo(
                name="meta-llama-3.1-8b",
                description="Meta Llama 3.1 8B",
                parameters="8B",
                context_window=128000,
                license="Llama 3.1",
                category="general",
                tags=["llama", "meta"]
            ),
            ModelInfo(
                name="ministral-3b",
                description="Ministral 3B",
                parameters="3B",
                context_window=32768,
                license="Apache 2.0",
                category="general",
                tags=["mistral", "apache-2.0", "small"]
            ),
            ModelInfo(
                name="mistral-medium-3",
                description="Mistral Medium 3",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["mistral", "medium"]
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
                name="openai-gpt-4.1",
                description="OpenAI GPT-4.1",
                parameters="Unknown",
                context_window=1000000,
                license="Proprietary",
                category="general",
                tags=["openai", "gpt-4"]
            ),
            ModelInfo(
                name="openai-gpt-4.1-mini",
                description="OpenAI GPT-4.1 Mini",
                parameters="Unknown",
                context_window=128000,
                license="Proprietary",
                category="general",
                tags=["openai", "gpt-4"]
            ),
            ModelInfo(
                name="openai-o1",
                description="OpenAI o1",
                parameters="Unknown",
                context_window=1000000,
                license="Proprietary",
                category="reasoning",
                tags=["openai", "o1"]
            ),
            ModelInfo(
                name="openai-o3",
                description="OpenAI o3",
                parameters="Unknown",
                context_window=1000000,
                license="Proprietary",
                category="reasoning",
                tags=["openai", "o3"]
            ),
            ModelInfo(
                name="openai-o4",
                description="OpenAI o4",
                parameters="Unknown",
                context_window=1000000,
                license="Proprietary",
                category="reasoning",
                tags=["openai", "o4"]
            ),
            ModelInfo(
                name="phi-4",
                description="Phi 4",
                parameters="14B",
                context_window=128000,
                license="MIT",
                category="general",
                tags=["phi", "microsoft"]
            ),
            ModelInfo(
                name="phi-4-mini",
                description="Phi 4 Mini",
                parameters="3.8B",
                context_window=128000,
                license="MIT",
                category="general",
                tags=["phi", "microsoft", "small"]
            ),
        ]
        
        return ProviderConfig(
            name="GitHub Models (Copilot)",
            category=ProviderCategory.LLM,
            provider_type=ProviderType.CLOUD,
            base_url=cls.BASE_URL,
            api_key_env_var="GITHUB_COPILOT_API_KEY",
            rate_limits=RateLimit(
                requests_per_minute=30,  # Estimated
                requests_per_day=100,  # Depends on subscription
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
        Initialize GitHub Models provider.
        
        Args:
            api_key: GitHub Copilot API key
        """
        if api_key is None:
            api_key = os.getenv("GITHUB_COPILOT_API_KEY")
        
        super().__init__(api_key=api_key)
        self.session = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API."""
        headers = super()._get_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
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
        Send a chat completion request to GitHub Copilot.
        
        Note: GitHub Copilot API may have different endpoints and formats.
        This is a placeholder implementation.
        """
        if not self._check_rate_limits():
            raise RateLimitExceededError("GitHub Models rate limit exceeded")
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # GitHub Copilot uses OpenAI-compatible format
        github_messages = []
        for msg in messages:
            if isinstance(msg, str):
                github_messages.append({"role": "user", "content": msg})
            else:
                github_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        payload = {
            "model": model,
            "messages": github_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 8192,
        }
        
        # GitHub Copilot may use different endpoints
        # This is a placeholder - actual implementation may vary
        url = f"{self.BASE_URL}/chat/completions"
        
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
                    error_msg = error_data.get("message", "Unknown error")
                    raise APIError(f"GitHub Copilot API error: {error_msg}")
                
                result = response.json()
                
                # Extract text from response
                if "choices" in result and len(result["choices"]) > 0:
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