"""
Example usage of OpenAI LLM Service.

This example demonstrates how to use the OpenAI LLM service for
voice-optimized conversational AI with function calling support.

**Validates: Requirements 13.1, 6.1, 6.2, 6.3**
"""

import asyncio
import os
from dotenv import load_dotenv

from src.voice_agent.services.openai_llm_service import OpenAILLMService
from src.voice_agent.services.llm_service import (
    LLMConfig,
    ConversationContext,
)


async def main():
    """Demonstrate OpenAI LLM service usage."""
    
    # Load environment variables
    load_dotenv()
    
    # Create LLM configuration
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are Sarah, a friendly medical receptionist helping patients schedule appointments.",
    )
    
    # Initialize OpenAI LLM service
    # API key will be loaded from OPENAI_API_KEY environment variable
    llm_service = OpenAILLMService(config)
    
    print("OpenAI LLM Service Example")
    print("=" * 50)
    print()
    
    # Example 1: Simple inquiry (no tool execution)
    print("Example 1: General Inquiry")
    print("-" * 50)
    
    context = ConversationContext(
        session_id="example_session_1",
        conversation_history=[],
    )
    
    transcript = "What are your office hours?"
    print(f"User: {transcript}")
    
    response = await llm_service.generate_response(transcript, context)
    
    print(f"Assistant: {response.text}")
    print(f"Intent: {response.intent.name}")
    print(f"Confidence: {response.confidence}")
    print(f"Tool Call: {response.tool_call}")
    print()
    
    # Example 2: Appointment booking (with tool execution)
    print("Example 2: Appointment Booking with Function Call")
    print("-" * 50)
    
    context = ConversationContext(
        session_id="example_session_2",
        conversation_history=[
            {"role": "assistant", "content": "Hi! How can I help you today?"},
        ],
    )
    
    transcript = "I'd like to schedule an appointment for tomorrow at 2pm for a checkup"
    print(f"User: {transcript}")
    
    response = await llm_service.generate_response(transcript, context)
    
    print(f"Assistant: {response.text}")
    print(f"Intent: {response.intent.name}")
    print(f"Confidence: {response.confidence}")
    
    if response.tool_call:
        print(f"Tool Call: {response.tool_call.tool_name}")
        print(f"Parameters: {response.tool_call.parameters}")
    
    print()
    
    # Example 3: Multi-turn conversation
    print("Example 3: Multi-turn Conversation")
    print("-" * 50)
    
    context = ConversationContext(
        session_id="example_session_3",
        conversation_history=[
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "I need to reschedule my appointment"},
            {"role": "assistant", "content": "I can help you with that. What's your appointment ID?"},
        ],
    )
    
    transcript = "It's appointment 12345, can we move it to Friday at 10am?"
    print(f"User: {transcript}")
    
    response = await llm_service.generate_response(transcript, context)
    
    print(f"Assistant: {response.text}")
    print(f"Intent: {response.intent.name}")
    
    if response.tool_call:
        print(f"Tool Call: {response.tool_call.tool_name}")
        print(f"Parameters: {response.tool_call.parameters}")
    
    print()
    
    # Example 4: Voice optimization demonstration
    print("Example 4: Voice Optimization")
    print("-" * 50)
    
    # Simulate a response with markdown that needs optimization
    markdown_text = """
    Here are **three options** for you:
    - Option 1: Monday at 9am
    - Option 2: Tuesday at 2pm
    - Option 3: Wednesday at 4pm
    
    Please let me know which works best!
    """
    
    optimized = llm_service.format_for_voice(markdown_text)
    
    print("Original (with markdown):")
    print(markdown_text)
    print()
    print("Voice-optimized:")
    print(optimized)
    print()
    
    # Example 5: Emotional tag extraction
    print("Example 5: Emotional Tag Extraction")
    print("-" * 50)
    
    emotional_responses = [
        "I'm sorry to hear you're not feeling well. I understand that must be frustrating.",
        "That's wonderful! I'm so excited to help you with that.",
        "I understand your concern. Would you like me to check available appointments?",
    ]
    
    for text in emotional_responses:
        tags = llm_service._extract_emotional_tags(text)
        print(f"Text: {text}")
        print(f"Emotional Tags: {tags}")
        print()
    
    print("=" * 50)
    print("Examples completed successfully!")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
        print()
        print("For demonstration purposes, the examples will show the structure")
        print("but won't make actual API calls.")
        print()
    
    asyncio.run(main())
