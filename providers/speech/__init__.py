"""
Speech Providers Package

Collection of free speech-to-text and text-to-speech API providers.
"""

from .groq_speech import GroqSpeechProvider
from .google_tts import GoogleTTSProvider
from .whisper_local import WhisperLocalProvider

__all__ = [
    "GroqSpeechProvider",
    "GoogleTTSProvider",
    "WhisperLocalProvider",
]