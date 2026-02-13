"""
Example usage of vLLM LLM Service.

This example demonstrates how to use the vLLM LLM service for high-performance
local inference with PagedAttention, continuous batching, and GPU resource management.

**Validates: Requirements 13.3, 9.4, 13.5**
"""

import asyncio
import logging

from src.voice_agent.services.vllm_llm_service import vLLMLLMService
from src.voice_agent.services.llm_service import LLMConfig, ConversationContext


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def basic_usage():
    """Demonstrate basic vLLM service usage."""
    logger.info("=== Basic vLLM Service Usage ===")
    
    # Create vLLM configuration
    config = LLMConfig(
        provider="vllm",
        model="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist named Sarah.",
    )
    
    # Initialize vLLM service with GPU resource management
    async with vLLMLLMService(
        config=config,
        base_url="http://localhost:8000",  # vLLM server endpoint
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        max_num_seqs=256,  # Support 256 concurrent sequences (continuous batching)
    ) as service:
        # Create conversation context
        context = ConversationContext(
            session_id="example_session_001",
            conversation_history=[],
        )
        
        # Generate response
        transcript = "I need to schedule an appointment for tomorrow at 2pm"
        logger.info("User: %s", transcript)
        
        response = await service.generate_response(transcript, context)
        
        logger.info("Assistant: %s", response.text)
        logger.info("Intent: %s (confidence: %.2f)", response.intent.name, response.intent.confidence)
        logger.info("Parameters: %s", response.parameters)
        
        if response.tool_call:
            logger.info("Tool call: %s with params %s", response.tool_call.tool_name, response.tool_call.parameters)
        
        logger.info("GPU metadata: memory_util=%.1f%%, max_seqs=%d",
                   response.metadata["gpu_memory_utilization"] * 100,
                   response.metadata["max_num_seqs"])


async def high_performance_configuration():
    """Demonstrate high-performance vLLM configuration."""
    logger.info("\n=== High-Performance vLLM Configuration ===")
    
    config = LLMConfig(
        provider="vllm",
        model="meta-llama/Llama-2-13b-chat-hf",  # Larger model
        temperature=0.7,
        max_tokens=200,
        system_prompt="You are a helpful assistant.",
    )
    
    # Configure for maximum throughput
    async with vLLMLLMService(
        config=config,
        base_url="http://localhost:8000",
        gpu_memory_utilization=0.95,  # Use 95% of GPU memory for larger model
        max_num_seqs=512,  # Support 512 concurrent sequences for high throughput
        max_model_len=4096,  # Set maximum context length
    ) as service:
        context = ConversationContext(session_id="perf_test_001")
        
        # Check server availability
        is_available = await service.check_server_availability()
        logger.info("vLLM server available: %s", is_available)
        
        if is_available:
            transcript = "What are your office hours?"
            response = await service.generate_response(transcript, context)
            logger.info("Response: %s", response.text)
            logger.info("Performance config: GPU=%.0f%%, max_seqs=%d, max_len=%s",
                       service.gpu_memory_utilization * 100,
                       service.max_num_seqs,
                       service.max_model_len)


async def with_fallback_service():
    """Demonstrate vLLM with fallback to cloud service."""
    logger.info("\n=== vLLM with Fallback Service ===")
    
    # Import OpenAI service as fallback
    from src.voice_agent.services.openai_llm_service import OpenAILLMService
    
    # Create fallback service (OpenAI)
    fallback_config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist.",
    )
    fallback_service = OpenAILLMService(fallback_config)
    
    # Create vLLM service with fallback
    vllm_config = LLMConfig(
        provider="vllm",
        model="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist.",
    )
    
    service = vLLMLLMService(
        config=vllm_config,
        base_url="http://localhost:8000",
        fallback_service=fallback_service,
    )
    
    context = ConversationContext(session_id="fallback_test_001")
    
    try:
        # This will use vLLM if available, otherwise fall back to OpenAI
        transcript = "I want to cancel my appointment"
        logger.info("User: %s", transcript)
        
        response = await service.generate_response(transcript, context)
        
        logger.info("Assistant: %s", response.text)
        logger.info("Provider used: %s", response.metadata.get("provider", "unknown"))
        
    finally:
        await service.close()


async def multi_turn_conversation():
    """Demonstrate multi-turn conversation with vLLM."""
    logger.info("\n=== Multi-Turn Conversation ===")
    
    config = LLMConfig(
        provider="vllm",
        model="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist.",
    )
    
    async with vLLMLLMService(
        config=config,
        base_url="http://localhost:8000",
        gpu_memory_utilization=0.9,
        max_num_seqs=256,
    ) as service:
        # Create context with conversation history
        context = ConversationContext(
            session_id="multi_turn_001",
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help you today?"},
                {"role": "user", "content": "I need to schedule an appointment"},
                {"role": "assistant", "content": "I'd be happy to help you schedule an appointment. What date works best for you?"},
            ],
        )
        
        # Continue conversation
        transcript = "How about next Monday at 2pm?"
        logger.info("User: %s", transcript)
        
        response = await service.generate_response(transcript, context)
        
        logger.info("Assistant: %s", response.text)
        logger.info("Intent: %s", response.intent.name)
        logger.info("Extracted parameters: %s", response.parameters)
        
        if response.emotional_tags:
            logger.info("Emotional tags: %s", response.emotional_tags)


async def voice_optimized_responses():
    """Demonstrate voice-optimized response generation."""
    logger.info("\n=== Voice-Optimized Responses ===")
    
    config = LLMConfig(
        provider="vllm",
        model="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a friendly medical receptionist.",
    )
    
    async with vLLMLLMService(
        config=config,
        base_url="http://localhost:8000",
    ) as service:
        context = ConversationContext(session_id="voice_opt_001")
        
        # Test various scenarios
        test_cases = [
            "I'm feeling anxious about my appointment",
            "That's great news!",
            "Can you help me reschedule?",
        ]
        
        for transcript in test_cases:
            logger.info("\nUser: %s", transcript)
            
            response = await service.generate_response(transcript, context)
            
            logger.info("Assistant: %s", response.text)
            
            # Check voice optimization
            assert "**" not in response.text, "Response contains markdown"
            assert "|" not in response.text, "Response contains tables"
            
            # Count sentences
            sentences = [s.strip() for s in response.text.split(".") if s.strip()]
            logger.info("Sentence count: %d (should be ≤3)", len(sentences))
            
            if response.emotional_tags:
                logger.info("Emotional tags: %s", response.emotional_tags)


async def gpu_resource_monitoring():
    """Demonstrate GPU resource configuration and monitoring."""
    logger.info("\n=== GPU Resource Monitoring ===")
    
    # Test different GPU configurations
    configs = [
        {"gpu_memory_utilization": 0.7, "max_num_seqs": 128, "desc": "Conservative"},
        {"gpu_memory_utilization": 0.9, "max_num_seqs": 256, "desc": "Balanced"},
        {"gpu_memory_utilization": 0.95, "max_num_seqs": 512, "desc": "Aggressive"},
    ]
    
    for gpu_config in configs:
        logger.info("\n--- %s Configuration ---", gpu_config["desc"])
        logger.info("GPU Memory: %.0f%%", gpu_config["gpu_memory_utilization"] * 100)
        logger.info("Max Sequences: %d", gpu_config["max_num_seqs"])
        
        config = LLMConfig(
            provider="vllm",
            model="meta-llama/Llama-2-7b-chat-hf",
            temperature=0.7,
            max_tokens=150,
        )
        
        service = vLLMLLMService(
            config=config,
            base_url="http://localhost:8000",
            gpu_memory_utilization=gpu_config["gpu_memory_utilization"],
            max_num_seqs=gpu_config["max_num_seqs"],
        )
        
        # Verify configuration
        assert service.gpu_memory_utilization == gpu_config["gpu_memory_utilization"]
        assert service.max_num_seqs == gpu_config["max_num_seqs"]
        
        logger.info("✓ Configuration verified")
        
        await service.close()


async def main():
    """Run all examples."""
    try:
        # Note: These examples require a running vLLM server
        # Start vLLM server with: python -m vllm.entrypoints.openai.api_server \
        #   --model meta-llama/Llama-2-7b-chat-hf \
        #   --gpu-memory-utilization 0.9 \
        #   --max-num-seqs 256
        
        logger.info("vLLM LLM Service Examples")
        logger.info("=" * 50)
        logger.info("Note: These examples require a running vLLM server at http://localhost:8000")
        logger.info("=" * 50)
        
        await basic_usage()
        await high_performance_configuration()
        await with_fallback_service()
        await multi_turn_conversation()
        await voice_optimized_responses()
        await gpu_resource_monitoring()
        
        logger.info("\n" + "=" * 50)
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error("Error running examples: %s", str(e))
        logger.info("\nMake sure vLLM server is running:")
        logger.info("  python -m vllm.entrypoints.openai.api_server \\")
        logger.info("    --model meta-llama/Llama-2-7b-chat-hf \\")
        logger.info("    --gpu-memory-utilization 0.9 \\")
        logger.info("    --max-num-seqs 256")


if __name__ == "__main__":
    asyncio.run(main())
