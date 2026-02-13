"""Session management with in-memory persistence and database-ready interface."""

from abc import ABC, abstractmethod
from datetime import datetime

from .session import ConversationTurn, Session


class SessionStore(ABC):
    """Abstract interface for session persistence.
    
    This interface defines the contract for session storage, allowing for
    different implementations (in-memory, database, etc.) without changing
    the SessionManager logic.
    """

    @abstractmethod
    def save(self, session: Session) -> None:
        """Save or update a session.
        
        Args:
            session: The session to save
        """
        pass

    @abstractmethod
    def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID.
        
        Args:
            session_id: The session identifier
            
        Returns:
            The session if found, None otherwise
        """
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if the session was deleted, False if not found
        """
        pass

    @abstractmethod
    def list_active_sessions(self) -> list[Session]:
        """List all active (not ended) sessions.
        
        Returns:
            List of active sessions
        """
        pass

    @abstractmethod
    def get_by_caller(self, caller_id: str) -> list[Session]:
        """Get all sessions for a specific caller.
        
        Args:
            caller_id: The caller identifier
            
        Returns:
            List of sessions for this caller
        """
        pass


class InMemorySessionStore(SessionStore):
    """In-memory implementation of session storage.
    
    This implementation stores sessions in memory using a dictionary.
    Suitable for MVP and testing, but data is lost on restart.
    """

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        self._sessions: dict[str, Session] = {}

    def save(self, session: Session) -> None:
        """Save or update a session in memory.
        
        Args:
            session: The session to save
        """
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Session | None:
        """Retrieve a session from memory.
        
        Args:
            session_id: The session identifier
            
        Returns:
            The session if found, None otherwise
        """
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete a session from memory.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if the session was deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_active_sessions(self) -> list[Session]:
        """List all active sessions in memory.
        
        Returns:
            List of active sessions
        """
        return [
            session for session in self._sessions.values()
            if session.is_active()
        ]

    def get_by_caller(self, caller_id: str) -> list[Session]:
        """Get all sessions for a specific caller from memory.
        
        Args:
            caller_id: The caller identifier
            
        Returns:
            List of sessions for this caller
        """
        return [
            session for session in self._sessions.values()
            if session.caller_id == caller_id
        ]

    def clear(self) -> None:
        """Clear all sessions from memory.
        
        This method is useful for testing and cleanup.
        """
        self._sessions.clear()


class SessionManager:
    """Manages conversation sessions and their lifecycle.
    
    The SessionManager provides high-level operations for creating, updating,
    and managing conversation sessions. It uses a SessionStore for persistence,
    allowing different storage backends to be plugged in.
    """

    def __init__(self, store: SessionStore | None = None) -> None:
        """Initialize the session manager.
        
        Args:
            store: Session storage implementation. Defaults to InMemorySessionStore.
        """
        self._store = store or InMemorySessionStore()

    def create_session(
        self,
        tenant_id: str,
        caller_id: str,
        session_id: str | None = None,
        metadata: dict | None = None
    ) -> Session:
        """Create a new session.
        
        Args:
            tenant_id: The tenant identifier
            caller_id: The caller identifier
            session_id: Optional custom session ID (auto-generated if not provided)
            metadata: Optional initial metadata
            
        Returns:
            The newly created session
        """
        session = Session(
            tenant_id=tenant_id,
            caller_id=caller_id,
            metadata=metadata or {}
        )

        if session_id:
            session.session_id = session_id

        self._store.save(session)
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID.
        
        Args:
            session_id: The session identifier
            
        Returns:
            The session if found, None otherwise
        """
        return self._store.get(session_id)

    def update_session(self, session: Session) -> None:
        """Update an existing session.
        
        Args:
            session: The session to update
        """
        self._store.save(session)

    def add_turn(
        self,
        session_id: str,
        speaker: str,
        transcript: str,
        intent: str | None = None,
        parameters: dict | None = None,
        tool_calls: list[str] | None = None,
        metadata: dict | None = None
    ) -> ConversationTurn | None:
        """Add a conversation turn to a session.
        
        Args:
            session_id: The session identifier
            speaker: Speaker identifier ('user' or 'agent')
            transcript: The transcribed or generated text
            intent: Optional classified intent
            parameters: Optional extracted parameters
            tool_calls: Optional list of tool names called
            metadata: Optional turn metadata
            
        Returns:
            The created conversation turn, or None if session not found
        """
        session = self._store.get(session_id)
        if not session:
            return None

        turn = ConversationTurn(
            speaker=speaker,
            transcript=transcript,
            intent=intent,
            parameters=parameters or {},
            tool_calls=tool_calls or [],
            metadata=metadata or {}
        )

        session.add_turn(turn)
        self._store.save(session)

        return turn

    def end_session(self, session_id: str) -> bool:
        """End a session and mark it as complete.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if the session was ended, False if not found
        """
        session = self._store.get(session_id)
        if not session:
            return False

        session.end_session()
        self._store.save(session)

        return True

    def cleanup_session(
        self,
        session_id: str,
        persist: bool = True
    ) -> dict | None:
        """Clean up session resources and optionally persist logs.
        
        This method performs cleanup operations when a session ends,
        including finalizing the session, generating a summary, and
        optionally persisting conversation logs for audit purposes.
        
        Args:
            session_id: The session identifier
            persist: Whether to persist the session (True) or delete it (False)
            
        Returns:
            Session summary dictionary, or None if session not found
        """
        session = self._store.get(session_id)
        if not session:
            return None

        # Ensure session is marked as ended
        if session.is_active():
            session.end_session()
            self._store.save(session)

        # Generate session summary
        summary = {
            "session_id": session.session_id,
            "tenant_id": session.tenant_id,
            "caller_id": session.caller_id,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "duration_seconds": (
                (session.end_time - session.start_time).total_seconds()
                if session.end_time else None
            ),
            "turn_count": session.get_turn_count(),
            "final_intent": session.current_intent,
            "extracted_parameters": session.extracted_parameters,
            "workflow_stage": session.workflow_stage,
            "transferred_to_human": session.metadata.get("transfer_requested", False),
            "transfer_reason": session.metadata.get("transfer_reason"),
            "metadata": session.metadata
        }

        # If not persisting, delete the session from storage
        if not persist:
            self._store.delete(session_id)

        return summary

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from storage.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if the session was deleted, False if not found
        """
        return self._store.delete(session_id)

    def get_active_sessions(self) -> list[Session]:
        """Get all currently active sessions.
        
        Returns:
            List of active sessions
        """
        return self._store.list_active_sessions()

    def get_caller_sessions(self, caller_id: str) -> list[Session]:
        """Get all sessions for a specific caller.
        
        Args:
            caller_id: The caller identifier
            
        Returns:
            List of sessions for this caller
        """
        return self._store.get_by_caller(caller_id)

    def update_workflow_stage(
        self,
        session_id: str,
        stage: str
    ) -> bool:
        """Update the workflow stage for a session.
        
        Args:
            session_id: The session identifier
            stage: The new workflow stage
            
        Returns:
            True if updated, False if session not found
        """
        session = self._store.get(session_id)
        if not session:
            return False

        session.workflow_stage = stage
        self._store.save(session)

        return True

    def get_conversation_history(
        self,
        session_id: str,
        limit: int | None = None
    ) -> list[ConversationTurn] | None:
        """Get conversation history for a session.
        
        Args:
            session_id: The session identifier
            limit: Optional limit on number of turns to return (most recent)
            
        Returns:
            List of conversation turns, or None if session not found
        """
        session = self._store.get(session_id)
        if not session:
            return None

        history = session.conversation_history

        if limit and limit > 0:
            return history[-limit:]

        return history

    def update_context(
        self,
        session_id: str,
        turn: ConversationTurn
    ) -> bool:
        """Update conversation context with a new turn.
        
        This method adds a conversation turn to the session and updates
        the session's extracted parameters and current intent based on
        the turn's content.
        
        Args:
            session_id: The session identifier
            turn: The conversation turn to add
            
        Returns:
            True if updated successfully, False if session not found
        """
        session = self._store.get(session_id)
        if not session:
            return False

        session.add_turn(turn)
        self._store.save(session)

        return True

    def get_context(self, session_id: str) -> dict | None:
        """Get the current conversation context for a session.
        
        Returns a dictionary containing the current conversation state,
        including conversation history, extracted parameters, current intent,
        and workflow stage.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Dictionary with context information, or None if session not found
        """
        session = self._store.get(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "tenant_id": session.tenant_id,
            "caller_id": session.caller_id,
            "start_time": session.start_time,
            "conversation_history": session.conversation_history,
            "extracted_parameters": session.extracted_parameters,
            "current_intent": session.current_intent,
            "workflow_stage": session.workflow_stage,
            "turn_count": session.get_turn_count(),
            "is_active": session.is_active(),
            "metadata": session.metadata
        }

    def transfer_to_human(
        self,
        session_id: str,
        reason: str,
        metadata: dict | None = None
    ) -> dict | None:
        """Initiate a transfer to a human agent.
        
        This method marks the session for human handoff, records the transfer
        reason, and returns transfer information that can be used by the
        telephony system to route the call.
        
        Args:
            session_id: The session identifier
            reason: Reason for the transfer (e.g., "user_request", "agent_limitation")
            metadata: Optional additional transfer metadata
            
        Returns:
            Dictionary with transfer information, or None if session not found
        """
        session = self._store.get(session_id)
        if not session:
            return None

        # Update session metadata with transfer information
        transfer_info = {
            "transfer_requested": True,
            "transfer_reason": reason,
            "transfer_timestamp": datetime.utcnow().isoformat(),
            "transfer_metadata": metadata or {}
        }

        session.metadata.update(transfer_info)
        session.workflow_stage = "transferring_to_human"

        self._store.save(session)

        # Return transfer information for telephony system
        return {
            "session_id": session.session_id,
            "tenant_id": session.tenant_id,
            "caller_id": session.caller_id,
            "reason": reason,
            "transfer_initiated": True,  # Add for backward compatibility
            "conversation_summary": {
                "turn_count": session.get_turn_count(),
                "current_intent": session.current_intent,
                "extracted_parameters": session.extracted_parameters,
                "last_turns": session.conversation_history[-5:] if len(session.conversation_history) >= 5 else session.conversation_history
            },
            "metadata": metadata or {}
        }
