"""
Example usage of LLM Provider Factory.

This example demonstrates how to use the LLM provider factory to create
different LLM service instances based on configuration, with automatic
health checking and failover support.

**Validates: Requirements 13.4, 13.6, 13.7**
"""

import asyncio
import os
from dotenv import load_dotenv

from src.voice_agent.services.llm_factory import LLMProviderFactory, ProviderHealth
from src.voice_agent.services.llm_service import LLMConfig, ConversationContext


async def main():
    """Demonstrate LLM provider factory usage."""
    
    # Load environment variables
    load_dotenv()
    
    print("=" * 60)
    print("LLM Provider Factory Example")
    print("=" * 60)
    
    # Initialize factory with configuration
    factory = LLMProviderFactory(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        ollama_base_url="http://localhost:11434",
        vllm_base_url="http://localhost:8000",
        enable_failover=True,  # Enable automatic failover
    )
    
    # Check health of all providers
    print("\n1. Checking provider health...")
    health_status = await factory.get_all_provider_health()
    for provider, health in health_status.items():
        print(f"   {provider}: {health.value}")
    
    # Example 1: Create OpenAI service
    print("\n2. Creating OpenAI LLM service...")
    openai_config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist.",
    )
    
    try:
        openai_service = await factory.create_llm_service(openai_config)
        print(f"   ✓ Created {openai_service.__class__.__name__}")
        print(f"   Model: {openai_service.config.model}")
        
        # Test OpenAI service
        context = ConversationContext(
            session_id="test-session-1",
            conversation_history=[],
        )
        
        response = await openai_service.generate_response(
            "I'd like to schedule an appointment for next Monday at 2pm",
            context,
        )
        
        print(f"   Intent: {response.intent.name}")
        print(f"   Response: {response.text}")
        if response.tool_call:
            print(f"   Tool: {response.tool_call.tool_name}")
            print(f"   Parameters: {response.tool_call.parameters}")
    
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Example 2: Create Ollama service with fallback
    print("\n3. Creating Ollama LLM service (with OpenAI fallback)...")
    ollama_config = LLMConfig(
        provider="ollama",
        model="llama2",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist.",
    )
    
    try:
        ollama_service = await factory.create_llm_service(ollama_config)
        print(f"   ✓ Created {ollama_service.__class__.__name__}")
        print(f"   Model: {ollama_service.config.model}")
        
        # Check if fallback is configured
        if hasattr(ollama_service, 'fallback_service') and ollama_service.fallback_service:
            print(f"   Fallback: {ollama_service.fallback_service.__class__.__name__}")
        
        # Note: Actual generation would require Ollama server running
        print("   (Skipping generation - requires Ollama server)")
    
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Example 3: Create vLLM service
    print("\n4. Creating vLLM service (with OpenAI fallback)...")
    vllm_config = LLMConfig(
        provider="vllm",
        model="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist.",
    )
    
    # Custom provider configuration
    provider_config = {
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 256,
        "fallback_model": "gpt-3.5-turbo",  # Custom fallback model
    }
    
    try:
        vllm_service = await factory.create_llm_service(vllm_config, provider_config)
        print(f"   ✓ Created {vllm_service.__class__.__name__}")
        print(f"   Model: {vllm_service.config.model}")
        
        # Check if fallback is configured
        if hasattr(vllm_service, 'fallback_service') and vllm_service.fallback_service:
            print(f"   Fallback: {vllm_service.fallback_service.__class__.__name__}")
            print(f"   Fallback Model: {vllm_service.fallback_service.config.model}")
        
        # Note: Actual generation would require vLLM server running
        print("   (Skipping generation - requires vLLM server)")
    
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Example 4: Demonstrate failover behavior
    print("\n5. Demonstrating automatic failover...")
    print("   When primary provider is unavailable, factory automatically")
    print("   creates fallback service (Ollama/vLLM → OpenAI)")
    
    # Clear health cache to force re-check
    factory.clear_health_cache()
    
    # Check individual provider health
    ollama_health = await factory.check_provider_health("ollama")
    print(f"   Ollama health: {ollama_health.value}")
    
    if ollama_health == ProviderHealth.UNAVAILABLE:
        print("   → Ollama unavailable, would use OpenAI fallback")
    else:
        print("   → Ollama available, would use primary provider")
    
    # Example 5: Factory without failover
    print("\n6. Creating factory without failover...")
    factory_no_failover = LLMProviderFactory(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        enable_failover=False,
    )
    
    print("   ✓ Factory created with failover disabled")
    print("   → Provider failures will raise errors instead of failing over")
    
    # Example 6: Concurrent service creation
    print("\n7. Creating multiple services concurrently...")
    try:
        services = await asyncio.gather(
            factory.create_llm_service(openai_config),
            factory.create_llm_service(ollama_config),
        )
        
        print(f"   ✓ Created {len(services)} services concurrently")
        for i, service in enumerate(services, 1):
            print(f"   Service {i}: {service.__class__.__name__}")
    
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
