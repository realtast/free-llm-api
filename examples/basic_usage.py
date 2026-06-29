"""
Basic Usage Examples

Simple examples of using the free LLM API providers.
"""

import asyncio
import os
from typing import Optional

from providers.llm.google_ai_studio import GoogleAIStudioProvider
from providers.llm.groq import GroqProvider
from providers.llm.mistral import MistralProvider
from providers.llm.openrouter import OpenRouterProvider
from providers.image.stable_diffusion import StableDiffusionProvider
from providers.image.google_gemini_image import GoogleGeminiImageProvider
from providers.speech.groq_speech import GroqSpeechProvider
from providers.embeddings.huggingface_embeddings import HuggingFaceEmbeddingsProvider


async def llm_example():
    """
    Example of using LLM providers.
    """
    print("=== LLM Examples ===")
    
    # Example 1: Google AI Studio
    print("\n1. Google AI Studio (Gemma 3)")
    try:
        # Get API key from environment or use None for testing
        google_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        if google_key:
            google = GoogleAIStudioProvider(api_key=google_key)
            response = await google.chat("gemma-3-27b-instruct", "What is the capital of France?")
            print(f"Response: {response}")
        else:
            print("Google AI Studio API key not found in environment")
    except Exception as e:
        print(f"Google AI Studio error: {e}")
    
    # Example 2: Groq
    print("\n2. Groq (Llama 3.1 8B)")
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            groq = GroqProvider(api_key=groq_key)
            response = await groq.chat("llama-3.1-8b-instant", "Explain quantum computing in simple terms")
            print(f"Response: {response}")
        else:
            print("Groq API key not found in environment")
    except Exception as e:
        print(f"Groq error: {e}")
    
    # Example 3: Mistral
    print("\n3. Mistral AI")
    try:
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if mistral_key:
            mistral = MistralProvider(api_key=mistral_key)
            response = await mistral.chat("mistral-small-2402", "What are the latest trends in AI?")
            print(f"Response: {response}")
        else:
            print("Mistral API key not found in environment")
    except Exception as e:
        print(f"Mistral error: {e}")
    
    # Example 4: OpenRouter
    print("\n4. OpenRouter")
    try:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            openrouter = OpenRouterProvider(api_key=openrouter_key)
            response = await openrouter.chat("llama-3.2-3b-instruct", "Tell me a joke")
            print(f"Response: {response}")
        else:
            print("OpenRouter API key not found in environment")
    except Exception as e:
        print(f"OpenRouter error: {e}")


async def image_example():
    """
    Example of using image generation providers.
    """
    print("\n=== Image Generation Examples ===")
    
    # Example 1: Stable Diffusion (local)
    print("\n1. Stable Diffusion (local)")
    try:
        # This requires a local Stable Diffusion server (e.g., ComfyUI)
        sd = StableDiffusionProvider(base_url="http://localhost:7860")
        image_data = await sd.generate("A beautiful sunset over mountains")
        print(f"Generated image data length: {len(image_data)} bytes")
        # In a real application, you would save this to a file
        # with open("sunset.png", "wb") as f:
        #     f.write(image_data)
    except Exception as e:
        print(f"Stable Diffusion error: {e}")
    
    # Example 2: Google Gemini Image
    print("\n2. Google Gemini Image")
    try:
        gemini_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        if gemini_key:
            gemini = GoogleGeminiImageProvider(api_key=gemini_key)
            image_data = await gemini.generate("A futuristic cityscape")
            print(f"Generated image data length: {len(image_data)} bytes")
        else:
            print("Google Gemini API key not found in environment")
    except Exception as e:
        print(f"Google Gemini Image error: {e}")


async def speech_example():
    """
    Example of using speech providers.
    """
    print("\n=== Speech Examples ===")
    
    # Example 1: Groq Speech (STT)
    print("\n1. Groq Speech (Speech-to-Text)")
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            speech = GroqSpeechProvider(api_key=groq_key)
            # Note: This would require an actual audio file
            # For demonstration, we'll just show the structure
            print("To transcribe audio: await speech.transcribe('audio.mp3')")
        else:
            print("Groq API key not found in environment")
    except Exception as e:
        print(f"Groq Speech error: {e}")


async def embedding_example():
    """
    Example of using embedding providers.
    """
    print("\n=== Embedding Examples ===")
    
    # Example 1: Hugging Face Embeddings (local)
    print("\n1. Hugging Face Embeddings (local)")
    try:
        # This requires a local embedding server
        embeddings = HuggingFaceEmbeddingsProvider(base_url="http://localhost:8000")
        embedding = await embeddings.embed("sentence-transformers/all-mpnet-base-v2", "Hello world")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First few values: {embedding[:5]}")
    except Exception as e:
        print(f"Hugging Face Embeddings error: {e}")


async def main():
    """Run all basic examples."""
    print("Free LLM API - Basic Usage Examples")
    print("=" * 50)
    
    await llm_example()
    await image_example()
    await speech_example()
    await embedding_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())