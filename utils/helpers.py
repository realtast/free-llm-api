"""
Helper Functions

Utility helper functions for the free LLM API providers.
"""

import re
import uuid
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def format_model_name(model_name: str) -> str:
    """
    Format a model name for consistent use.
    
    Args:
        model_name: Original model name
        
    Returns:
        str: Formatted model name
    """
    if not model_name:
        return ""
    
    # Convert to lowercase
    formatted = model_name.lower()
    
    # Replace common separators with hyphens
    formatted = re.sub(r'[\s/_\-\.]', '-', formatted)
    
    # Remove duplicate hyphens
    formatted = re.sub(r'-+', '-', formatted)
    
    # Remove leading/trailing hyphens
    formatted = formatted.strip('-')
    
    return formatted


def parse_size_string(size_str: str) -> Tuple[int, int]:
    """
    Parse a size string (e.g., "1024x1024") into width and height.
    
    Args:
        size_str: Size string in format "WxH" or "widthxheight"
        
    Returns:
        Tuple[int, int]: (width, height)
        
    Raises:
        ValueError: If the size string is invalid
    """
    if not size_str:
        return 1024, 1024
    
    # Try to parse WxH format
    match = re.match(r'^(\d+)x(\d+)$', size_str.lower())
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height
    
    # Try to parse as single number (square)
    try:
        size = int(size_str)
        return size, size
    except ValueError:
        pass
    
    # Default to 1024x1024
    logger.warning(f"Invalid size string: {size_str}, defaulting to 1024x1024")
    return 1024, 1024


def validate_api_key(api_key: str, provider: str = "") -> bool:
    """
    Validate an API key format.
    
    Args:
        api_key: API key to validate
        provider: Provider name for context
        
    Returns:
        bool: True if the API key appears valid
    """
    if not api_key:
        return False
    
    # Check for reasonable length (most API keys are 20+ characters)
    if len(api_key) < 10:
        logger.warning(f"API key for {provider} appears too short")
        return False
    
    # Check for common patterns
    # This is a basic check - actual validation requires API calls
    common_patterns = [
        r'^[a-zA-Z0-9]{20,}$',  # Alphanumeric
        r'^[a-zA-Z0-9_-]{30,}$',  # Alphanumeric with underscores/hyphens
        r'^sk-[a-zA-Z0-9]{20,}$',  # OpenAI-style
        r'^xox[baprs]-[a-zA-Z0-9]{20,}$',  # Slack-style
    ]
    
    for pattern in common_patterns:
        if re.match(pattern, api_key):
            return True
    
    # If it doesn't match common patterns, it might still be valid
    # We'll be lenient and accept it
    return True


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text for API requests.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length (optional)
        
    Returns:
        str: Sanitized text
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'[\s]+', ' ', text)
    
    # Trim
    text = text.strip()
    
    # Apply max length
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with optional suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        str: Truncated text
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    # Account for suffix length
    if len(suffix) >= max_length:
        return suffix[:max_length]
    
    truncated = text[:max_length - len(suffix)] + suffix
    return truncated


def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a unique ID.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of the random part
        
    Returns:
        str: Unique ID
    """
    # Generate random part
    random_part = uuid.uuid4().hex[:length]
    
    # Combine with prefix
    if prefix:
        return f"{prefix}_{random_part}"
    
    return random_part


def generate_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Generate a hash of text.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (default: sha256)
        
    Returns:
        str: Hexadecimal hash string
    """
    if not text:
        return ""
    
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()


def timestamp() -> str:
    """
    Get current timestamp as ISO format string.
    
    Returns:
        str: ISO format timestamp
    """
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        str: Human-readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_tokens(num_tokens: int) -> str:
    """
    Format token count into human-readable string.
    
    Args:
        num_tokens: Number of tokens
        
    Returns:
        str: Human-readable token count string
    """
    if num_tokens < 1000:
        return f"{num_tokens} tokens"
    elif num_tokens < 1000000:
        return f"{num_tokens / 1000:.1f}K tokens"
    else:
        return f"{num_tokens / 1000000:.1f}M tokens"


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text that may contain JSON.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Dict[str, Any]: Extracted JSON or None if not found
    """
    if not text:
        return None
    
    # Try to find JSON in the text
    # Look for text between { and }
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            import json
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to parse the whole text as JSON
    try:
        import json
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    return None


def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary.
    
    Args:
        dictionary: Dictionary to get value from
        key: Key to look up
        default: Default value if key not found
        
    Returns:
        Any: Value from dictionary or default
    """
    if not dictionary:
        return default
    
    try:
        return dictionary[key]
    except (KeyError, TypeError):
        return default