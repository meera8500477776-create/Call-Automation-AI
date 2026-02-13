"""
Example usage of Ollama LLM Service.

This example demonstrates how to use the Ollama LLM service for local
LLM deployment with GPU acceleration and fallback handling.

**Validates: Requirements 13.2, 13.5, 7.6**
"""

import asyncio
import logging
from src.voice_agent.services.ollama_llm_service import OllamaLLMService
from src.voice_agent.services.openai_llm_service import OpenAILLMService
from src.voice_agent.services.llm_service import LLMConfig, ConversationContext


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Example: Basic Ollama LLM service usage."""
    print("\n=== Example 1: Basic Ollama Usage ===\n")
    
    # Create Ollama configuration
    config = LLMConfig(
        provider="ollama",
        model="llama2",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist named Sarah.",
    )
    
    # Initialize Ollama service
    async with OllamaLLMService(config=config) as service:
        # Check if model is available
        is_available = await service.check_model_availability()
        print(f"Model available: {is_available}")
        
        if not is_available:
            print("Note: Ollama model not available. Please ensure Ollama is running.")
            print("Install: https://ollama.ai")
            print("Run: ollama pull llama2")
            return
        
        # Create conversation context
        context = ConversationContext(
            session_id="example-session-1",
            conversation_history=[],
        )
        
        # Generate response
        transcript = "I need to schedule an appointment for tomorrow at 2pm"
        response = await service.generate_response(transcript, context)
        
        print(f"User: {transcript}")
        print(f"Assistant: {response.text}")
        print(f"Intent: {response.intent.name}")
        print(f"Parameters: {response.parameters}")
        if response.tool_call:
            print(f"Tool Call: {response.tool_call.tool_name}")


async def example_with_fallback():
    """Example: Ollama with OpenAI fallback."""
    print("\n=== Example 2: Ollama with Fallback ===\n")
    
    # Create OpenAI fallback service
    fallback_config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist.",
    )
    
    # Note: Requires OPENAI_API_KEY environment variable
    try:
        fallback_service = OpenAILLMService(config=fallback_config)
    except Exception as e:
        print(f"Could not initialize fallback service: {e}")
        print("Set OPENAI_API_KEY environment variable to use fallback")
        fallback_service = None
    
    # Create Ollama configuration with fallback
    ollama_config = LLMConfig(
        provider="ollama",
        model="llama2",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist.",
    )
    
    # Initialize Ollama service with fallback
    service = OllamaLLMService(
        config=ollama_config,
        fallback_service=fallback_service,
    )
    
    # Create conversation context
    context = ConversationContext(
        session_id="example-session-2",
        conversation_history=[],
    )
    
    # Generate response (will use fallback if Ollama unavailable)
    transcript = "Can I reschedule my appointment?"
    try:
        response = await service.generate_response(transcript, context)
        
        print(f"User: {transcript}")
        print(f"Assistant: {response.text}")
        print(f"Provider: {response.metadata.get('provider', 'unknown')}")
        print(f"Intent: {response.intent.name}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await service.close()


async def example_gpu_configuration():
    """Example: Ollama with GPU acceleration."""
    print("\n=== Example 3: GPU Acceleration ===\n")
    
    # Create Ollama configuration
    config = LLMConfig(
        provider="ollama",
        model="llama2",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful assistant.",
    )
    
    # Initialize with GPU enabled (default)
    async with OllamaLLMService(
        config=config,
        gpu_enabled=True,  # Use all available GPUs
    ) as service:
        print(f"GPU enabled: {service.gpu_enabled}")
        
        # Check model availability
        is_available = await service.check_model_availability()
        if not is_available:
            print("Model not available")
            return
        
        # Create conversation context
        context = ConversationContext(
            session_id="example-session-3",
            conversation_history=[],
        )
        
        # Generate response
        transcript = "Hello, how are you?"
        response = await service.generate_response(transcript, context)
        
        print(f"User: {transcript}")
        print(f"Assistant: {response.text}")


async def example_multi_turn_conversation():
    """Example: Multi-turn conversation with context."""
    print("\n=== Example 4: Multi-Turn Conversation ===\n")
    
    # Create Ollama configuration
    config = LLMConfig(
        provider="ollama",
        model="llama2",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist.",
    )
    
    async with OllamaLLMService(config=config) as service:
        # Check model availability
        is_available = await service.check_model_availability()
        if not is_available:
            print("Model not available")
            return
        
        # Create conversation context
        context = ConversationContext(
            session_id="example-session-4",
            conversation_history=[],
        )
        
        # Turn 1
        transcript1 = "Hello"
        response1 = await service.generate_response(transcript1, context)
        print(f"User: {transcript1}")
        print(f"Assistant: {response1.text}\n")
        
        # Update context
        context.conversation_history.append({
            "role": "user",
            "content": transcript1,
        })
        context.conversation_history.append({
            "role": "assistant",
            "content": response1.text,
        })
        
        # Turn 2
        transcript2 = "I need to schedule an appointment"
        response2 = await service.generate_response(transcript2, context)
        print(f"User: {transcript2}")
        print(f"Assistant: {response2.text}")
        print(f"Intent: {response2.intent.name}")
        
        # Update context
        context.conversation_history.append({
            "role": "user",
            "content": transcript2,
        })
        context.conversation_history.append({
            "role": "assistant",
            "content": response2.text,
        })
        
        # Turn 3
        transcript3 = "Tomorrow at 2pm would be great"
        response3 = await service.generate_response(transcript3, context)
        print(f"\nUser: {transcript3}")
        print(f"Assistant: {response3.text}")
        print(f"Parameters: {response3.parameters}")


async def example_custom_base_url():
    """Example: Ollama with custom base URL."""
    print("\n=== Example 5: Custom Base URL ===\n")
    
    # Create Ollama configuration
    config = LLMConfig(
        provider="ollama",
        model="llama2",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful assistant.",
    )
    
    # Initialize with custom base URL (e.g., remote Ollama server)
    async with OllamaLLMService(
        config=config,
        base_url="http://localhost:11434",  # Default, can be changed
    ) as service:
        print(f"Base URL: {service.base_url}")
        
        # Check model availability
        is_available = await service.check_model_availability()
        print(f"Model available: {is_available}")


async def main():
    """Run all examples."""
    try:
        await example_basic_usage()
        await example_with_fallback()
        await example_gpu_configuration()
        await example_multi_turn_conversation()
        await example_custom_base_url()
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        logger.error("Error running examples: %s", str(e), exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
