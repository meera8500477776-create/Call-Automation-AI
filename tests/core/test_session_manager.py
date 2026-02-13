"""Unit tests for SessionManager and SessionStore implementations."""


from voice_agent.core import (
    ConversationTurn,
    InMemorySessionStore,
    Session,
    SessionManager,
)


class TestInMemorySessionStore:
    """Tests for InMemorySessionStore."""

    def test_save_and_get_session(self):
        """Test saving and retrieving a session."""
        store = InMemorySessionStore()
        session = Session(tenant_id="tenant1", caller_id="+15551234567")

        store.save(session)
        retrieved = store.get(session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == session.session_id
        assert retrieved.tenant_id == session.tenant_id
        assert retrieved.caller_id == session.caller_id

    def test_get_nonexistent_session(self):
        """Test retrieving a session that doesn't exist."""
        store = InMemorySessionStore()

        retrieved = store.get("nonexistent_id")

        assert retrieved is None

    def test_update_existing_session(self):
        """Test updating an existing session."""
        store = InMemorySessionStore()
        session = Session(tenant_id="tenant1", caller_id="+15551234567")

        store.save(session)

        # Modify the session
        session.workflow_stage = "collecting_info"
        store.save(session)

        retrieved = store.get(session.session_id)
        assert retrieved.workflow_stage == "collecting_info"

    def test_delete_session(self):
        """Test deleting a session."""
        store = InMemorySessionStore()
        session = Session(tenant_id="tenant1", caller_id="+15551234567")

        store.save(session)
        assert store.get(session.session_id) is not None

        result = store.delete(session.session_id)
        assert result is True
        assert store.get(session.session_id) is None

    def test_delete_nonexistent_session(self):
        """Test deleting a session that doesn't exist."""
        store = InMemorySessionStore()

        result = store.delete("nonexistent_id")

        assert result is False

    def test_list_active_sessions(self):
        """Test listing active sessions."""
        store = InMemorySessionStore()

        session1 = Session(tenant_id="tenant1", caller_id="+15551234567")
        session2 = Session(tenant_id="tenant1", caller_id="+15551234568")
        session3 = Session(tenant_id="tenant1", caller_id="+15551234569")

        store.save(session1)
        store.save(session2)
        store.save(session3)

        # End one session
        session2.end_session()
        store.save(session2)

        active = store.list_active_sessions()

        assert len(active) == 2
        assert session1 in active
        assert session3 in active
        assert session2 not in active

    def test_get_by_caller(self):
        """Test getting sessions by caller ID."""
        store = InMemorySessionStore()

        session1 = Session(tenant_id="tenant1", caller_id="+15551234567")
        session2 = Session(tenant_id="tenant1", caller_id="+15551234567")
        session3 = Session(tenant_id="tenant1", caller_id="+15551234568")

        store.save(session1)
        store.save(session2)
        store.save(session3)

        caller_sessions = store.get_by_caller("+15551234567")

        assert len(caller_sessions) == 2
        assert session1 in caller_sessions
        assert session2 in caller_sessions
        assert session3 not in caller_sessions

    def test_clear(self):
        """Test clearing all sessions."""
        store = InMemorySessionStore()

        session1 = Session(tenant_id="tenant1", caller_id="+15551234567")
        session2 = Session(tenant_id="tenant1", caller_id="+15551234568")

        store.save(session1)
        store.save(session2)

        assert len(store.list_active_sessions()) == 2

        store.clear()

        assert len(store.list_active_sessions()) == 0
        assert store.get(session1.session_id) is None
        assert store.get(session2.session_id) is None


class TestSessionManager:
    """Tests for SessionManager."""

    def test_create_session(self):
        """Test creating a new session."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        assert session is not None
        assert session.tenant_id == "tenant1"
        assert session.caller_id == "+15551234567"
        assert session.is_active()

    def test_create_session_with_custom_id(self):
        """Test creating a session with a custom ID."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567",
            session_id="custom_session_id"
        )

        assert session.session_id == "custom_session_id"

    def test_create_session_with_metadata(self):
        """Test creating a session with initial metadata."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567",
            metadata={"call_sid": "CA123456"}
        )

        assert session.metadata == {"call_sid": "CA123456"}

    def test_get_session(self):
        """Test retrieving a session."""
        manager = SessionManager()

        created = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        retrieved = manager.get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_session(self):
        """Test retrieving a session that doesn't exist."""
        manager = SessionManager()

        retrieved = manager.get_session("nonexistent_id")

        assert retrieved is None

    def test_update_session(self):
        """Test updating a session."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        session.workflow_stage = "collecting_info"
        manager.update_session(session)

        retrieved = manager.get_session(session.session_id)
        assert retrieved.workflow_stage == "collecting_info"

    def test_add_turn(self):
        """Test adding a turn to a session."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        turn = manager.add_turn(
            session_id=session.session_id,
            speaker="user",
            transcript="I need an appointment",
            intent="schedule_appointment",
            parameters={"date": "2024-01-15"},
            tool_calls=["search_appointments"],
            metadata={"confidence": 0.95}
        )

        assert turn is not None
        assert turn.speaker == "user"
        assert turn.transcript == "I need an appointment"
        assert turn.intent == "schedule_appointment"
        assert turn.parameters == {"date": "2024-01-15"}
        assert turn.tool_calls == ["search_appointments"]
        assert turn.metadata == {"confidence": 0.95}

        # Verify the turn was added to the session
        retrieved = manager.get_session(session.session_id)
        assert len(retrieved.conversation_history) == 1
        assert retrieved.conversation_history[0] == turn

    def test_add_turn_to_nonexistent_session(self):
        """Test adding a turn to a session that doesn't exist."""
        manager = SessionManager()

        turn = manager.add_turn(
            session_id="nonexistent_id",
            speaker="user",
            transcript="Hello"
        )

        assert turn is None

    def test_end_session(self):
        """Test ending a session."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        assert session.is_active()

        result = manager.end_session(session.session_id)

        assert result is True

        retrieved = manager.get_session(session.session_id)
        assert not retrieved.is_active()

    def test_end_nonexistent_session(self):
        """Test ending a session that doesn't exist."""
        manager = SessionManager()

        result = manager.end_session("nonexistent_id")

        assert result is False

    def test_delete_session(self):
        """Test deleting a session."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        result = manager.delete_session(session.session_id)

        assert result is True
        assert manager.get_session(session.session_id) is None

    def test_delete_nonexistent_session(self):
        """Test deleting a session that doesn't exist."""
        manager = SessionManager()

        result = manager.delete_session("nonexistent_id")

        assert result is False

    def test_get_active_sessions(self):
        """Test getting all active sessions."""
        manager = SessionManager()

        session1 = manager.create_session(tenant_id="tenant1", caller_id="+15551234567")
        session2 = manager.create_session(tenant_id="tenant1", caller_id="+15551234568")
        session3 = manager.create_session(tenant_id="tenant1", caller_id="+15551234569")

        # End one session
        manager.end_session(session2.session_id)

        active = manager.get_active_sessions()

        assert len(active) == 2
        session_ids = [s.session_id for s in active]
        assert session1.session_id in session_ids
        assert session3.session_id in session_ids
        assert session2.session_id not in session_ids

    def test_get_caller_sessions(self):
        """Test getting sessions for a specific caller."""
        manager = SessionManager()

        session1 = manager.create_session(tenant_id="tenant1", caller_id="+15551234567")
        session2 = manager.create_session(tenant_id="tenant1", caller_id="+15551234567")
        session3 = manager.create_session(tenant_id="tenant1", caller_id="+15551234568")

        caller_sessions = manager.get_caller_sessions("+15551234567")

        assert len(caller_sessions) == 2
        session_ids = [s.session_id for s in caller_sessions]
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids
        assert session3.session_id not in session_ids

    def test_update_workflow_stage(self):
        """Test updating the workflow stage."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        assert session.workflow_stage == "initial"

        result = manager.update_workflow_stage(
            session_id=session.session_id,
            stage="collecting_info"
        )

        assert result is True

        retrieved = manager.get_session(session.session_id)
        assert retrieved.workflow_stage == "collecting_info"

    def test_update_workflow_stage_nonexistent_session(self):
        """Test updating workflow stage for a session that doesn't exist."""
        manager = SessionManager()

        result = manager.update_workflow_stage(
            session_id="nonexistent_id",
            stage="collecting_info"
        )

        assert result is False

    def test_get_conversation_history(self):
        """Test getting conversation history."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        turn1 = manager.add_turn(session.session_id, "user", "Hello")
        turn2 = manager.add_turn(session.session_id, "agent", "Hi")
        turn3 = manager.add_turn(session.session_id, "user", "I need help")

        history = manager.get_conversation_history(session.session_id)

        assert history is not None
        assert len(history) == 3
        assert history[0] == turn1
        assert history[1] == turn2
        assert history[2] == turn3

    def test_get_conversation_history_with_limit(self):
        """Test getting conversation history with a limit."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        turn1 = manager.add_turn(session.session_id, "user", "Hello")
        turn2 = manager.add_turn(session.session_id, "agent", "Hi")
        turn3 = manager.add_turn(session.session_id, "user", "I need help")

        history = manager.get_conversation_history(session.session_id, limit=2)

        assert history is not None
        assert len(history) == 2
        assert history[0] == turn2
        assert history[1] == turn3

    def test_get_conversation_history_nonexistent_session(self):
        """Test getting conversation history for a session that doesn't exist."""
        manager = SessionManager()

        history = manager.get_conversation_history("nonexistent_id")

        assert history is None

    def test_custom_store(self):
        """Test using a custom session store."""
        custom_store = InMemorySessionStore()
        manager = SessionManager(store=custom_store)

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        # Verify the session is in the custom store
        assert custom_store.get(session.session_id) is not None


class TestSessionManagerNewMethods:
    """Tests for new SessionManager methods: update_context, get_context, transfer_to_human, cleanup_session."""

    def test_update_context(self):
        """Test updating conversation context with a new turn."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        turn = ConversationTurn(
            speaker="user",
            transcript="I need an appointment",
            intent="schedule_appointment",
            parameters={"date": "2024-01-15"}
        )

        result = manager.update_context(session.session_id, turn)

        assert result is True

        # Verify the turn was added
        retrieved = manager.get_session(session.session_id)
        assert len(retrieved.conversation_history) == 1
        assert retrieved.conversation_history[0] == turn
        assert retrieved.current_intent == "schedule_appointment"
        assert retrieved.extracted_parameters == {"date": "2024-01-15"}

    def test_update_context_nonexistent_session(self):
        """Test updating context for a session that doesn't exist."""
        manager = SessionManager()

        turn = ConversationTurn(
            speaker="user",
            transcript="Hello"
        )

        result = manager.update_context("nonexistent_id", turn)

        assert result is False

    def test_get_context(self):
        """Test retrieving current conversation context."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567",
            metadata={"call_sid": "CA123456"}
        )

        # Add some turns
        manager.add_turn(session.session_id, "user", "Hello")
        manager.add_turn(session.session_id, "agent", "Hi, how can I help?")

        context = manager.get_context(session.session_id)

        assert context is not None
        assert context["session_id"] == session.session_id
        assert context["tenant_id"] == "tenant1"
        assert context["caller_id"] == "+15551234567"
        assert context["turn_count"] == 2
        assert context["is_active"] is True
        assert len(context["conversation_history"]) == 2
        assert context["metadata"] == {"call_sid": "CA123456"}

    def test_get_context_nonexistent_session(self):
        """Test getting context for a session that doesn't exist."""
        manager = SessionManager()

        context = manager.get_context("nonexistent_id")

        assert context is None

    def test_get_context_with_parameters_and_intent(self):
        """Test getting context with extracted parameters and intent."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        manager.add_turn(
            session.session_id,
            "user",
            "I need an appointment on January 15th",
            intent="schedule_appointment",
            parameters={"date": "2024-01-15"}
        )

        context = manager.get_context(session.session_id)

        assert context["current_intent"] == "schedule_appointment"
        assert context["extracted_parameters"] == {"date": "2024-01-15"}

    def test_transfer_to_human(self):
        """Test initiating a transfer to human agent."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        # Add some conversation history
        manager.add_turn(session.session_id, "user", "I need help")
        manager.add_turn(session.session_id, "agent", "I can help with that")
        manager.add_turn(session.session_id, "user", "I want to speak to a person")

        transfer_info = manager.transfer_to_human(
            session_id=session.session_id,
            reason="user_request",
            metadata={"priority": "high"}
        )

        assert transfer_info is not None
        assert transfer_info["session_id"] == session.session_id
        assert transfer_info["tenant_id"] == "tenant1"
        assert transfer_info["caller_id"] == "+15551234567"
        assert transfer_info["reason"] == "user_request"
        assert transfer_info["metadata"] == {"priority": "high"}
        assert transfer_info["conversation_summary"]["turn_count"] == 3

        # Verify session was updated
        retrieved = manager.get_session(session.session_id)
        assert retrieved.workflow_stage == "transferring_to_human"
        assert retrieved.metadata["transfer_requested"] is True
        assert retrieved.metadata["transfer_reason"] == "user_request"
        assert "transfer_timestamp" in retrieved.metadata

    def test_transfer_to_human_nonexistent_session(self):
        """Test transferring a session that doesn't exist."""
        manager = SessionManager()

        transfer_info = manager.transfer_to_human(
            session_id="nonexistent_id",
            reason="user_request"
        )

        assert transfer_info is None

    def test_transfer_to_human_conversation_summary(self):
        """Test that transfer includes conversation summary with last turns."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        # Add more than 5 turns
        for i in range(7):
            manager.add_turn(session.session_id, "user", f"Message {i}")

        transfer_info = manager.transfer_to_human(
            session_id=session.session_id,
            reason="agent_limitation"
        )

        # Should only include last 5 turns
        assert len(transfer_info["conversation_summary"]["last_turns"]) == 5
        assert transfer_info["conversation_summary"]["turn_count"] == 7

    def test_cleanup_session_with_persist(self):
        """Test cleaning up a session with persistence."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        # Add some conversation
        manager.add_turn(session.session_id, "user", "Hello")
        manager.add_turn(session.session_id, "agent", "Hi")

        summary = manager.cleanup_session(session.session_id, persist=True)

        assert summary is not None
        assert summary["session_id"] == session.session_id
        assert summary["tenant_id"] == "tenant1"
        assert summary["caller_id"] == "+15551234567"
        assert summary["turn_count"] == 2
        assert summary["duration_seconds"] is not None
        assert summary["transferred_to_human"] is False

        # Session should still exist
        assert manager.get_session(session.session_id) is not None

    def test_cleanup_session_without_persist(self):
        """Test cleaning up a session without persistence (deletion)."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        summary = manager.cleanup_session(session.session_id, persist=False)

        assert summary is not None
        assert summary["session_id"] == session.session_id

        # Session should be deleted
        assert manager.get_session(session.session_id) is None

    def test_cleanup_session_nonexistent(self):
        """Test cleaning up a session that doesn't exist."""
        manager = SessionManager()

        summary = manager.cleanup_session("nonexistent_id")

        assert summary is None

    def test_cleanup_session_with_transfer(self):
        """Test cleanup summary includes transfer information."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        # Initiate transfer
        manager.transfer_to_human(
            session_id=session.session_id,
            reason="user_request"
        )

        summary = manager.cleanup_session(session.session_id)

        assert summary["transferred_to_human"] is True
        assert summary["transfer_reason"] == "user_request"
        assert summary["workflow_stage"] == "transferring_to_human"

    def test_cleanup_session_auto_ends_active_session(self):
        """Test that cleanup automatically ends an active session."""
        manager = SessionManager()

        session = manager.create_session(
            tenant_id="tenant1",
            caller_id="+15551234567"
        )

        assert session.is_active()

        summary = manager.cleanup_session(session.session_id)

        assert summary is not None
        assert summary["end_time"] is not None

        # Verify session is no longer active
        retrieved = manager.get_session(session.session_id)
        assert not retrieved.is_active()
