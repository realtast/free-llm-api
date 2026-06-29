"""
Advanced Usage Examples

More advanced examples of using the free LLM API providers.
"""

import asyncio
import os
from typing import List, Dict, Any

from providers.llm.google_ai_studio import GoogleAIStudioProvider
from providers.llm.groq import GroqProvider
from providers.llm.mistral import MistralProvider
from providers.llm.openrouter import OpenRouterProvider
from utils.rate_limiter import RateLimiter, RateLimitConfig
from utils.retry_logic import retry_with_backoff, RetryConfig
from utils.helpers import sanitize_text, truncate_text, generate_id


async def multi_provider_example():
    """
    Example of using multiple providers with fallback.
    """
    print("=== Multi-Provider Example ===")
    
    providers = []
    
    # Initialize available providers
    if os.getenv("GOOGLE_AI_STUDIO_API_KEY"):
        providers.append(("Google AI Studio", GoogleAIStudioProvider(os.getenv("GOOGLE_AI_STUDIO_API_KEY"))))
    
    if os.getenv("GROQ_API_KEY"):
        providers.append(("Groq", GroqProvider(os.getenv("GROQ_API_KEY"))))
    
    if os.getenv("MISTRAL_API_KEY"):
        providers.append(("Mistral", MistralProvider(os.getenv("MISTRAL_API_KEY"))))
    
    if os.getenv("OPENROUTER_API_KEY"):
        providers.append(("OpenRouter", OpenRouterProvider(os.getenv("OPENROUTER_API_KEY"))))
    
    if not providers:
        print("No API keys found. Please set your API keys in environment variables.")
        return
    
    # Define the prompt
    prompt = "What are the key differences between Python and JavaScript?"
    
    # Try each provider in order
    for provider_name, provider in providers:
        try:
            print(f"\nTrying {provider_name}...")
            
            # Get available models
            models = provider.get_available_models()
            if not models:
                print(f"No models available for {provider_name}")
                continue
            
            # Use the first available model
            model = models[0]
            print(f"Using model: {model}")
            
            # Get response
            response = await provider.chat(model, prompt)
            
            if response:
                print(f"\n{provider_name} ({model}) response:")
                print("-" * 50)
                print(response)
                print("-" * 50)
                break  # Success, stop trying other providers
            else:
                print(f"Empty response from {provider_name}")
                
        except Exception as e:
            print(f"Error with {provider_name}: {e}")
    else:
        print("All providers failed. Please check your API keys and network connection.")


async def rate_limiting_example():
    """
    Example of rate limiting with providers.
    """
    print("\n=== Rate Limiting Example ===")
    
    # Create a rate limiter
    config = RateLimitConfig(
        requests_per_minute=10,
        requests_per_hour=100
    )
    limiter = RateLimiter(config)
    
    # Test rate limiting
    print("Testing rate limiting...")
    
    for i in range(15):
        # Check if we can make the request
        can_request = await limiter.check_rate_limit("request", 1)
        
        if can_request:
            await limiter.record_request("request", 1)
            print(f"Request {i+1}: Allowed")
        else:
            print(f"Request {i+1}: Rate limited")
            break
    
    print("Rate limiting test completed")


async def error_handling_example():
    """
    Example of error handling with retry logic.
    """
    print("\n=== Error Handling Example ===")
    
    # Create a retry configuration
    retry_config = RetryConfig(
        max_retries=3,
        delay=1.0,
        exponential_backoff=True,
        max_delay=10.0
    )
    
    async def unreliable_api_call():
        """Simulate an unreliable API call."""
        # This will fail the first few times
        if unreliable_api_call.attempt < 3:
            unreliable_api_call.attempt += 1
            raise ConnectionError("Simulated connection error")
        return "Success! The API call worked."
    
    unreliable_api_call.attempt = 0
    
    try:
        # Use retry logic
        result = await retry_with_backoff(unreliable_api_call, retry_config)
        print(f"Result after retries: {result}")
    except Exception as e:
        print(f"Failed after all retries: {e}")


async def batch_processing_example():
    """
    Example of batch processing with multiple prompts.
    """
    print("\n=== Batch Processing Example ===")
    
    # List of prompts to process
    prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "What are the types of AI?",
        "How does deep learning work?",
        "What is the future of AI?"
    ]
    
    # Use Groq if available (has good rate limits)
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("Groq API key not found. Skipping batch processing example.")
        return
    
    try:
        groq = GroqProvider(api_key=groq_key)
        model = "llama-3.1-8b-instant"  # Good for batch processing
        
        print(f"Processing {len(prompts)} prompts with {model}...")
        
        # Process prompts in parallel (with rate limiting)
        tasks = []
        for i, prompt in enumerate(prompts):
            # Add small delay between requests to respect rate limits
            await asyncio.sleep(0.1)
            task = asyncio.create_task(groq.chat(model, prompt))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Display results
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"\nPrompt {i+1}: {prompt}")
            if isinstance(result, Exception):
                print(f"Error: {result}")
            else:
                print(f"Response: {truncate_text(result, 100)}")
        
    except Exception as e:
        print(f"Batch processing error: {e}")


async def conversation_example():
    """
    Example of maintaining a conversation with context.
    """
    print("\n=== Conversation Example ===")
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("Groq API key not found. Skipping conversation example.")
        return
    
    try:
        groq = GroqProvider(api_key=groq_key)
        model = "llama-3.1-8b-instant"
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Hello! How are you today?"}
        ]
        
        print("Starting conversation...")
        
        # First message
        response = await groq.chat(model, messages)
        print(f"AI: {response}")
        
        # Add AI response to conversation
        messages.append({"role": "assistant", "content": response})
        
        # User follow-up
        user_message = "What can you help me with?"
        messages.append({"role": "user", "content": user_message})
        print(f"User: {user_message}")
        
        # Get AI response
        response = await groq.chat(model, messages)
        print(f"AI: {response}")
        
        # Add to conversation
        messages.append({"role": "assistant", "content": response})
        
        print("Conversation completed!")
        
    except Exception as e:
        print(f"Conversation error: {e}")


async def main():
    """Run all advanced examples."""
    print("Free LLM API - Advanced Usage Examples")
    print("=" * 50)
    
    await multi_provider_example()
    await rate_limiting_example()
    await error_handling_example()
    await batch_processing_example()
    await conversation_example()
    
    print("\n" + "=" * 50)
    print("Advanced examples completed!")


if __name__ == "__main__":
    asyncio.run(main())