"""
Settings

Configuration settings for the free LLM API providers.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Configuration settings."""
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_buffer: float = 0.1  # Use 10% buffer below limits
    
    # Retry logic
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0  # Seconds between retries
    exponential_backoff: bool = True
    
    # Timeouts
    timeout_seconds: float = 120.0
    connect_timeout: float = 30.0
    
    # Logging
    debug_mode: bool = False
    log_requests: bool = True
    log_level: str = "INFO"
    
    # API keys (loaded from environment or config file)
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Provider-specific settings
    provider_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize settings after creation."""
        # Load API keys from environment
        self._load_api_keys_from_env()
        
        # Set up logging
        self._setup_logging()
    
    def _load_api_keys_from_env(self):
        """Load API keys from environment variables."""
        # List of all possible API key environment variables
        api_key_vars = [
            # LLM Providers
            "GOOGLE_AI_STUDIO_API_KEY",
            "GROQ_API_KEY",
            "MISTRAL_API_KEY",
            "OPENROUTER_API_KEY",
            "CEREBRAS_API_KEY",
            "COHERE_API_KEY",
            "CLOUDFLARE_API_KEY",
            "CLOUDFLARE_ACCOUNT_ID",
            "HUGGINGFACE_API_KEY",
            "NVIDIA_API_KEY",
            "VERCEL_AI_API_KEY",
            "GITHUB_COPILOT_API_KEY",
            
            # Image Providers
            "GOOGLE_GEMINI_API_KEY",
            "ADOBE_FIREFLY_API_KEY",
            "MIDJOURNEY_API_KEY",
            "OPENAI_API_KEY",
            "IMAGEN_API_KEY",
            
            # Local Deployment
            "STABLE_DIFFUSION_BASE_URL",
            "FLUX_BASE_URL",
            "WHISPER_BASE_URL",
            "HUGGINGFACE_EMBEDDINGS_BASE_URL",
        ]
        
        for var in api_key_vars:
            value = os.getenv(var)
            if value:
                self.api_keys[var] = value
    
    def _setup_logging(self):
        """Set up logging based on settings."""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_from_file(self, config_path: str):
        """
        Load settings from a YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            if config_data:
                # Update settings from file
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                    elif key == "api_keys":
                        self.api_keys.update(value)
                    elif key == "provider_settings":
                        self.provider_settings.update(value)
                
                logger.info(f"Loaded configuration from {config_path}")
                
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_to_file(self, config_path: str):
        """
        Save settings to a YAML configuration file.
        
        Args:
            config_path: Path to save the YAML configuration file
        """
        try:
            config_data = {
                "rate_limit_enabled": self.rate_limit_enabled,
                "rate_limit_buffer": self.rate_limit_buffer,
                "retry_on_failure": self.retry_on_failure,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "exponential_backoff": self.exponential_backoff,
                "timeout_seconds": self.timeout_seconds,
                "connect_timeout": self.connect_timeout,
                "debug_mode": self.debug_mode,
                "log_requests": self.log_requests,
                "log_level": self.log_level,
                "api_keys": self.api_keys,
                "provider_settings": self.provider_settings,
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get an API key by name.
        
        Args:
            key_name: Name of the API key
            
        Returns:
            str: API key value or None if not found
        """
        return self.api_keys.get(key_name)
    
    def set_api_key(self, key_name: str, value: str):
        """
        Set an API key.
        
        Args:
            key_name: Name of the API key
            value: API key value
        """
        self.api_keys[key_name] = value
    
    def get_provider_setting(self, provider: str, setting: str, default: Any = None) -> Any:
        """
        Get a provider-specific setting.
        
        Args:
            provider: Provider name
            setting: Setting name
            default: Default value if not found
            
        Returns:
            Any: Setting value or default
        """
        provider_settings = self.provider_settings.get(provider, {})
        return provider_settings.get(setting, default)
    
    def set_provider_setting(self, provider: str, setting: str, value: Any):
        """
        Set a provider-specific setting.
        
        Args:
            provider: Provider name
            setting: Setting name
            value: Setting value
        """
        if provider not in self.provider_settings:
            self.provider_settings[provider] = {}
        self.provider_settings[provider][setting] = value


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def configure_settings(config_path: Optional[str] = None):
    """
    Configure settings from a file or use defaults.
    
    Args:
        config_path: Optional path to YAML configuration file
    """
    global settings
    
    if config_path:
        settings.load_from_file(config_path)
    
    return settings