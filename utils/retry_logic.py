"""
Retry Logic

Retry utilities for API requests.
"""

import asyncio
import time
import random
from typing import Callable, Any, Optional, List, Type, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    delay: float = 1.0
    exponential_backoff: bool = True
    max_delay: float = 60.0
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    retryable_status_codes: List[int] = field(default_factory=list)
    retry_on_timeout: bool = True
    
    def __post_init__(self):
        """Initialize default retryable exceptions."""
        if not self.retryable_exceptions:
            # Common retryable exceptions
            import httpx
            self.retryable_exceptions = [
                ConnectionError,
                TimeoutError,
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
            ]
        
        if not self.retryable_status_codes:
            # Common retryable status codes
            self.retryable_status_codes = [
                408,  # Request Timeout
                429,  # Too Many Requests
                500,  # Internal Server Error
                502,  # Bad Gateway
                503,  # Service Unavailable
                504,  # Gateway Timeout
            ]


async def retry_with_backoff(
    func: Callable[..., Any],
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs
) -> Any:
    """
    Execute a function with retry logic and exponential backoff.
    
    Args:
        func: Function to execute
        config: Retry configuration
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Any: Result of the function
        
    Raises:
        Exception: If all retries fail
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
            
        except Exception as e:
            last_exception = e
            
            # Check if this is a retryable exception
            if not config._is_retryable_exception(e):
                raise
            
            # Check if we should retry
            if attempt >= config.max_retries:
                raise
            
            # Calculate delay
            delay = config._calculate_delay(attempt)
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            
            # Wait before retrying
            await asyncio.sleep(delay)
    
    # This should never be reached, but just in case
    raise last_exception


async def retry_with_custom_condition(
    func: Callable[..., Any],
    should_retry: Callable[[Exception, int], bool],
    max_retries: int = 3,
    delay: float = 1.0,
    exponential_backoff: bool = True,
    max_delay: float = 60.0,
    *args,
    **kwargs
) -> Any:
    """
    Execute a function with custom retry condition.
    
    Args:
        func: Function to execute
        should_retry: Function that takes (exception, attempt) and returns bool
        max_retries: Maximum number of retries
        delay: Initial delay in seconds
        exponential_backoff: Whether to use exponential backoff
        max_delay: Maximum delay in seconds
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Any: Result of the function
        
    Raises:
        Exception: If all retries fail or condition not met
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
            
        except Exception as e:
            last_exception = e
            
            # Check if we should retry
            if not should_retry(e, attempt):
                raise
            
            # Check if we've exhausted retries
            if attempt >= max_retries:
                raise
            
            # Calculate delay
            if exponential_backoff:
                current_delay = delay * (2 ** attempt)
            else:
                current_delay = delay
            
            current_delay = min(current_delay, max_delay)
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay:.2f}s...")
            
            # Wait before retrying
            await asyncio.sleep(current_delay)
    
    # This should never be reached
    raise last_exception


class Retryable:
    """
    Decorator for retryable functions.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize the retryable decorator.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
    
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorate a function to be retryable.
        
        Args:
            func: Function to decorate
            
        Returns:
            Callable: Decorated function
        """
        async def wrapper(*args, **kwargs) -> Any:
            return await retry_with_backoff(func, self.config, *args, **kwargs)
        
        return wrapper


# Add helper methods to RetryConfig

def _is_retryable_exception(self, exception: Exception) -> bool:
    """Check if an exception is retryable."""
    # Check if it's in our list of retryable exceptions
    for retryable_exc in self.retryable_exceptions:
        if isinstance(exception, retryable_exc):
            return True
    
    # Check for status code in HTTP exceptions
    import httpx
    if isinstance(exception, httpx.HTTPStatusError):
        if exception.response.status_code in self.retryable_status_codes:
            return True
    
    return False


def _calculate_delay(self, attempt: int) -> float:
    """Calculate delay for a retry attempt."""
    if self.exponential_backoff:
        delay = self.delay * (2 ** attempt)
    else:
        delay = self.delay
    
    # Add some jitter to prevent thundering herd
    jitter = random.uniform(0, 0.1 * delay)
    delay = delay + jitter
    
    return min(delay, self.max_delay)


# Monkey patch the methods
RetryConfig._is_retryable_exception = _is_retryable_exception
RetryConfig._calculate_delay = _calculate_delay