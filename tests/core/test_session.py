"""Unit tests for Session and ConversationTurn models."""

from datetime import datetime

import pytest

from voice_agent.core import ConversationTurn, Session


class TestConversationTurn:
    """Tests for ConversationTurn model."""

    def test_create_turn_with_required_fields(self):
        """Test creating a turn with only required fields."""
        turn = ConversationTurn(
            speaker="user",
            transcript="I need to schedule an appointment"
        )
        
        assert turn.speaker == "user"
        assert turn.transcript == "I need to schedule an appointment"
        assert turn.turn_id is not None
        assert isinstance(turn.timestamp, datetime)
        assert turn.intent is None
        assert turn.parameters == {}
        assert turn.tool_calls == []
        assert turn.metadata == {}

    def test_create_turn_with_all_fields(self):
        """Test creating a turn with all fields populated."""
        turn = ConversationTurn(
            speaker="agent",
            transcript="I can help you schedule an appointment",
            intent="schedule_appointment",
            parameters={"date": "2024-01-15", "time": "10:00"},
            tool_calls=["search_appointments"],
            metadata={"confidence": 0.95}
        )
        
        assert turn.speaker == "agent"
        assert turn.transcript == "I can help you schedule an appointment"
        assert turn.intent == "schedule_appointment"
        assert turn.parameters == {"date": "2024-01-15", "time": "10:00"}
        assert turn.tool_calls == ["search_appointments"]
        assert turn.metadata == {"confidence": 0.95}

    def test_turn_id_is_unique(self):
        """Test that each turn gets a unique ID."""
        turn1 = ConversationTurn(speaker="user", transcript="Hello")
        turn2 = ConversationTurn(speaker="user", transcript="Hello")
        
        assert turn1.turn_id != turn2.turn_id

    def test_turn_timestamp_is_set(self):
        """Test that timestamp is automatically set."""
        before = datetime.utcnow()
        turn = ConversationTurn(speaker="user", transcript="Hello")
        after = datetime.utcnow()
        
        assert before <= turn.timestamp <= after

    def test_turn_forbids_extra_fields(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValueError):
            ConversationTurn(
                speaker="user",
                transcript="Hello",
                extra_field="not allowed"
            )


class TestSession:
    """Tests for Session model."""

    def test_create_session_with_required_fields(self):
        """Test creating a session with only required fields."""
        session = Session(
            tenant_id="healthcare_clinic_001",
            caller_id="+15551234567"
        )
        
        assert session.tenant_id == "healthcare_clinic_001"
        assert session.caller_id == "+15551234567"
        assert session.session_id is not None
        assert isinstance(session.start_time, datetime)
        assert session.end_time is None
        assert session.conversation_history == []
        assert session.extracted_parameters == {}
        assert session.current_intent is None
        assert session.workflow_stage == "initial"
        assert session.metadata == {}

    def test_create_session_with_all_fields(self):
        """Test creating a session with all fields populated."""
        session = Session(
            tenant_id="healthcare_clinic_001",
            caller_id="+15551234567",
            extracted_parameters={"patient_name": "John Doe"},
            current_intent="schedule_appointment",
            workflow_stage="collecting_info",
            metadata={"call_sid": "CA123456"}
        )
        
        assert session.extracted_parameters == {"patient_name": "John Doe"}
        assert session.current_intent == "schedule_appointment"
        assert session.workflow_stage == "collecting_info"
        assert session.metadata == {"call_sid": "CA123456"}

    def test_session_id_is_unique(self):
        """Test that each session gets a unique ID."""
        session1 = Session(tenant_id="tenant1", caller_id="+15551234567")
        session2 = Session(tenant_id="tenant1", caller_id="+15551234567")
        
        assert session1.session_id != session2.session_id

    def test_add_turn_to_session(self):
        """Test adding a conversation turn to a session."""
        session = Session(tenant_id="tenant1", caller_id="+15551234567")
        turn = ConversationTurn(
            speaker="user",
            transcript="I need an appointment",
            intent="schedule_appointment",
            parameters={"date": "2024-01-15"}
        )
        
        session.add_turn(turn)
        
        assert len(session.conversation_history) == 1
        assert session.conversation_history[0] == turn
        assert session.current_intent == "schedule_appointment"
        assert session.extracted_parameters["date"] == "2024-01-15"

    def test_add_multiple_turns(self):
        """Test adding multiple turns to a session."""
        session = Session(tenant_id="tenant1", caller_id="+15551234567")
        
        turn1 = ConversationTurn(speaker="user", transcript="Hello")
        turn2 = ConversationTurn(speaker="agent", transcript="Hi, how can I help?")
        turn3 = ConversationTurn(speaker="user", transcript="I need an appointment")
        
        session.add_turn(turn1)
        session.add_turn(turn2)
        session.add_turn(turn3)
        
        assert len(session.conversation_history) == 3
        assert session.conversation_history[0] == turn1
        assert session.conversation_history[1] == turn2
        assert session.conversation_history[2] == turn3

    def test_add_turn_updates_intent(self):
        """Test that adding a turn with an intent updates the session's current intent."""
        session = Session(tenant_id="tenant1", caller_id="+15551234567")
        
        turn1 = ConversationTurn(
            speaker="user",
            transcript="Hello",
            intent="greeting"
        )
        session.add_turn(turn1)
        assert session.current_intent == "greeting"
        
        turn2 = ConversationTurn(
            speaker="user",
            transcript="I need an appointment",
            intent="schedule_appointment"
        )
        session.add_turn(turn2)
        assert session.current_intent == "schedule_appointment"

    def test_add_turn_merges_parameters(self):
        """Test that parameters from turns are merged into session parameters."""
        session = Session(tenant_id="tenant1", caller_id="+15551234567")
        
        turn1 = ConversationTurn(
            speaker="user",
            transcript="I need an appointment on January 15th",
            parameters={"date": "2024-01-15"}
        )
        session.add_turn(turn1)
        assert session.extracted_parameters == {"date": "2024-01-15"}
        
        turn2 = ConversationTurn(
            speaker="user",
            transcript="At 10 AM",
            parameters={"time": "10:00"}
        )
        session.add_turn(turn2)
        assert session.extracted_parameters == {"date": "2024-01-15", "time": "10:00"}

    def test_end_session(self):
        """Test ending a session."""
        session = Session(tenant_id="tenant1", caller_id="+15551234567")
        
        assert session.end_time is None
        assert session.is_active()
        
        before = datetime.utcnow()
        session.end_session()
        after = datetime.utcnow()
        
        assert session.end_time is not None
        assert before <= session.end_time <= after
        assert not session.is_active()

    def test_end_session_idempotent(self):
        """Test that ending a session multiple times doesn't change the end time."""
        session = Session(tenant_id="tenant1", caller_id="+15551234567")
        
        session.end_session()
        first_end_time = session.end_time
        
        session.end_session()
        second_end_time = session.end_time
        
        assert first_end_time == second_end_time

    def test_is_active(self):
        """Test the is_active method."""
        session = Session(tenant_id="tenant1", caller_id="+15551234567")
        
        assert session.is_active()
        
        session.end_session()
        
        assert not session.is_active()

    def test_get_turn_count(self):
        """Test getting the turn count."""
        session = Session(tenant_id="tenant1", caller_id="+15551234567")
        
        assert session.get_turn_count() == 0
        
        session.add_turn(ConversationTurn(speaker="user", transcript="Hello"))
        assert session.get_turn_count() == 1
        
        session.add_turn(ConversationTurn(speaker="agent", transcript="Hi"))
        assert session.get_turn_count() == 2

    def test_get_last_turn(self):
        """Test getting the last turn."""
        session = Session(tenant_id="tenant1", caller_id="+15551234567")
        
        assert session.get_last_turn() is None
        
        turn1 = ConversationTurn(speaker="user", transcript="Hello")
        session.add_turn(turn1)
        assert session.get_last_turn() == turn1
        
        turn2 = ConversationTurn(speaker="agent", transcript="Hi")
        session.add_turn(turn2)
        assert session.get_last_turn() == turn2

    def test_session_forbids_extra_fields(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValueError):
            Session(
                tenant_id="tenant1",
                caller_id="+15551234567",
                extra_field="not allowed"
            )
