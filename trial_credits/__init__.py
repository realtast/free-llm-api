"""
Trial Credits Package

Providers with trial credits (not completely free but offer generous trial credits).
"""

from .providers_with_credits import (
    TRIAL_CREDIT_PROVIDERS,
    get_provider_by_name,
    get_providers_by_credit_amount,
    get_providers_by_duration,
)

__all__ = [
    "TRIAL_CREDIT_PROVIDERS",
    "get_provider_by_name",
    "get_providers_by_credit_amount",
    "get_providers_by_duration",
]