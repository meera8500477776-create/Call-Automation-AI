"""Session and conversation data models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """Represents a single turn in a conversation.
    
    A turn captures one exchange in the conversation, including the speaker,
    transcript, intent classification, extracted parameters, and any tool calls made.
    """

    turn_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this conversation turn"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this turn occurred"
    )
    speaker: str = Field(
        ...,
        description="Speaker identifier: 'user' or 'agent'"
    )
    transcript: str = Field(
        ...,
        description="Transcribed or generated text for this turn"
    )
    intent: Optional[str] = Field(
        default=None,
        description="Classified intent name (e.g., 'schedule_appointment')"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted parameters from the utterance"
    )
    tool_calls: List[str] = Field(
        default_factory=list,
        description="List of tool names called during this turn"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (latency metrics, confidence scores, etc.)"
    )

    model_config = {"extra": "forbid"}


class Session(BaseModel):
    """Represents a complete conversation session.
    
    A session tracks the entire lifecycle of a call, including conversation history,
    extracted parameters across turns, current workflow state, and session metadata.
    """

    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this session"
    )
    tenant_id: str = Field(
        ...,
        description="Tenant identifier for this session"
    )
    caller_id: str = Field(
        ...,
        description="Caller phone number or identifier"
    )
    start_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the session started"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="When the session ended (None if still active)"
    )
    conversation_history: List[ConversationTurn] = Field(
        default_factory=list,
        description="Ordered list of conversation turns"
    )
    extracted_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters extracted and accumulated across turns"
    )
    current_intent: Optional[str] = Field(
        default=None,
        description="Current active intent being processed"
    )
    workflow_stage: str = Field(
        default="initial",
        description="Current stage in the conversation workflow"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional session metadata (call_sid, transfer_reason, etc.)"
    )

    model_config = {"extra": "forbid"}

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn to the session history.
        
        Args:
            turn: The conversation turn to add
        """
        self.conversation_history.append(turn)
        
        # Update current intent if the turn has one
        if turn.intent:
            self.current_intent = turn.intent
        
        # Merge turn parameters into session parameters
        if turn.parameters:
            self.extracted_parameters.update(turn.parameters)

    def end_session(self) -> None:
        """Mark the session as ended."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()

    def is_active(self) -> bool:
        """Check if the session is currently active.
        
        Returns:
            True if the session is active (not ended), False otherwise
        """
        return self.end_time is None

    def get_turn_count(self) -> int:
        """Get the total number of turns in this session.
        
        Returns:
            Number of conversation turns
        """
        return len(self.conversation_history)

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent conversation turn.
        
        Returns:
            The last conversation turn, or None if no turns exist
        """
        return self.conversation_history[-1] if self.conversation_history else None
