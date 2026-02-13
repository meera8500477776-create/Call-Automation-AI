"""Property-based tests for session management.

**Validates: Requirements 10.1**

This module contains property-based tests using Hypothesis to verify universal
properties of the session management system across all valid inputs.
"""

import pytest
from hypothesis import given, settings, strategies as st

from voice_agent.core import ConversationTurn, Session, SessionManager


# Hypothesis strategies for generating valid session data

@st.composite
def conversation_turn_strategy(draw):
    """Generate valid ConversationTurn instances."""
    speaker = draw(st.sampled_from(["user", "agent"]))
    transcript = draw(st.text(min_size=1, max_size=200))
    intent = draw(st.one_of(
        st.none(),
        st.sampled_from([
            "schedule_appointment",
            "reschedule_appointment",
            "cancel_appointment",
            "general_inquiry",
            "transfer_to_human"
        ])
    ))
    
    # Generate parameters based on intent
    parameters = {}
    if intent == "schedule_appointment":
        if draw(st.booleans()):
            parameters["date"] = draw(st.text(min_size=1, max_size=20))
        if draw(st.booleans()):
            parameters["time"] = draw(st.text(min_size=1, max_size=20))
        if draw(st.booleans()):
            parameters["reason"] = draw(st.text(min_size=1, max_size=50))
    elif intent == "reschedule_appointment":
        if draw(st.booleans()):
            parameters["appointment_id"] = draw(st.text(min_size=1, max_size=20))
        if draw(st.booleans()):
            parameters["new_date"] = draw(st.text(min_size=1, max_size=20))
    
    tool_calls = draw(st.lists(
        st.sampled_from([
            "search_appointments",
            "book_appointment",
            "cancel_appointment",
            "get_patient_info"
        ]),
        max_size=3
    ))
    
    metadata = draw(st.dictionaries(
        st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        st.one_of(
            st.text(max_size=50),
            st.floats(min_value=0.0, max_value=1.0),
            st.integers(min_value=0, max_value=1000),
            st.booleans()
        ),
        max_size=5
    ))
    
    return ConversationTurn(
        speaker=speaker,
        transcript=transcript,
        intent=intent,
        parameters=parameters,
        tool_calls=tool_calls,
        metadata=metadata
    )


@pytest.mark.property
class TestSessionContextPreservation:
    """Property 2: Session context preservation.
    
    **Validates: Requirements 10.1**
    
    For any sequence of conversation turns, retrieving context should return
    all turns in order.
    """

    @settings(max_examples=100, deadline=None)
    @given(st.lists(conversation_turn_strategy(), min_size=1, max_size=20))
    def test_context_preserves_all_turns_in_order(self, turns):
        """Test that get_context returns all turns in the order they were added.
        
        **Validates: Requirements 10.1**
        """
        # Create a session
        manager = SessionManager()
        session = manager.create_session(
            tenant_id="test_tenant",
            caller_id="+15551234567"
        )
        
        # Add all turns to the session
        for turn in turns:
            manager.update_context(session.session_id, turn)
        
        # Retrieve context
        context = manager.get_context(session.session_id)
        
        # Verify all turns are present
        assert context is not None
        assert len(context["conversation_history"]) == len(turns)
        
        # Verify turns are in the correct order
        for i, (original_turn, retrieved_turn) in enumerate(
            zip(turns, context["conversation_history"])
        ):
            assert retrieved_turn.speaker == original_turn.speaker
            assert retrieved_turn.transcript == original_turn.transcript
            assert retrieved_turn.intent == original_turn.intent
            assert retrieved_turn.parameters == original_turn.parameters
            assert retrieved_turn.tool_calls == original_turn.tool_calls
            assert retrieved_turn.metadata == original_turn.metadata

    @settings(max_examples=100, deadline=None)
    @given(st.lists(conversation_turn_strategy(), min_size=1, max_size=20))
    def test_context_turn_count_matches_added_turns(self, turns):
        """Test that turn_count in context matches the number of added turns.
        
        **Validates: Requirements 10.1**
        """
        # Create a session
        manager = SessionManager()
        session = manager.create_session(
            tenant_id="test_tenant",
            caller_id="+15551234567"
        )
        
        # Add all turns
        for turn in turns:
            manager.update_context(session.session_id, turn)
        
        # Retrieve context
        context = manager.get_context(session.session_id)
        
        # Verify turn count
        assert context["turn_count"] == len(turns)

    @settings(max_examples=100, deadline=None)
    @given(st.lists(conversation_turn_strategy(), min_size=1, max_size=20))
    def test_context_preserves_turn_identity(self, turns):
        """Test that each turn maintains its unique identity through context operations.
        
        **Validates: Requirements 10.1**
        """
        # Create a session
        manager = SessionManager()
        session = manager.create_session(
            tenant_id="test_tenant",
            caller_id="+15551234567"
        )
        
        # Add all turns and collect their IDs
        turn_ids = []
        for turn in turns:
            manager.update_context(session.session_id, turn)
            turn_ids.append(turn.turn_id)
        
        # Retrieve context
        context = manager.get_context(session.session_id)
        
        # Verify each turn has a unique ID
        retrieved_turn_ids = [turn.turn_id for turn in context["conversation_history"]]
        assert len(retrieved_turn_ids) == len(set(retrieved_turn_ids))
        
        # Verify turn IDs match
        assert retrieved_turn_ids == turn_ids

    @settings(max_examples=100, deadline=None)
    @given(
        st.lists(conversation_turn_strategy(), min_size=1, max_size=10),
        st.lists(conversation_turn_strategy(), min_size=1, max_size=10)
    )
    def test_context_preserves_order_across_multiple_updates(self, turns_batch1, turns_batch2):
        """Test that context preserves order when turns are added in multiple batches.
        
        **Validates: Requirements 10.1**
        """
        # Create a session
        manager = SessionManager()
        session = manager.create_session(
            tenant_id="test_tenant",
            caller_id="+15551234567"
        )
        
        # Add first batch of turns
        for turn in turns_batch1:
            manager.update_context(session.session_id, turn)
        
        # Add second batch of turns
        for turn in turns_batch2:
            manager.update_context(session.session_id, turn)
        
        # Retrieve context
        context = manager.get_context(session.session_id)
        
        # Verify all turns are present in order
        all_turns = turns_batch1 + turns_batch2
        assert len(context["conversation_history"]) == len(all_turns)
        
        for original_turn, retrieved_turn in zip(all_turns, context["conversation_history"]):
            assert retrieved_turn.speaker == original_turn.speaker
            assert retrieved_turn.transcript == original_turn.transcript

    @settings(max_examples=100, deadline=None)
    @given(st.lists(conversation_turn_strategy(), min_size=1, max_size=20))
    def test_context_preserves_parameters_across_turns(self, turns):
        """Test that extracted parameters are accumulated correctly across turns.
        
        **Validates: Requirements 10.1**
        """
        # Create a session
        manager = SessionManager()
        session = manager.create_session(
            tenant_id="test_tenant",
            caller_id="+15551234567"
        )
        
        # Track expected accumulated parameters
        expected_parameters = {}
        
        # Add all turns and accumulate parameters
        for turn in turns:
            manager.update_context(session.session_id, turn)
            expected_parameters.update(turn.parameters)
        
        # Retrieve context
        context = manager.get_context(session.session_id)
        
        # Verify accumulated parameters
        assert context["extracted_parameters"] == expected_parameters

    @settings(max_examples=100, deadline=None)
    @given(st.lists(conversation_turn_strategy(), min_size=1, max_size=20))
    def test_context_preserves_latest_intent(self, turns):
        """Test that context preserves the most recent intent from turns.
        
        **Validates: Requirements 10.1**
        """
        # Create a session
        manager = SessionManager()
        session = manager.create_session(
            tenant_id="test_tenant",
            caller_id="+15551234567"
        )
        
        # Add all turns
        for turn in turns:
            manager.update_context(session.session_id, turn)
        
        # Find the last turn with an intent
        expected_intent = None
        for turn in reversed(turns):
            if turn.intent is not None:
                expected_intent = turn.intent
                break
        
        # Retrieve context
        context = manager.get_context(session.session_id)
        
        # Verify current intent matches the last non-None intent
        assert context["current_intent"] == expected_intent

    @settings(max_examples=100, deadline=None)
    @given(st.lists(conversation_turn_strategy(), min_size=0, max_size=20))
    def test_context_handles_empty_conversation(self, turns):
        """Test that context correctly handles sessions with no or few turns.
        
        **Validates: Requirements 10.1**
        """
        # Create a session
        manager = SessionManager()
        session = manager.create_session(
            tenant_id="test_tenant",
            caller_id="+15551234567"
        )
        
        # Add turns (may be empty list)
        for turn in turns:
            manager.update_context(session.session_id, turn)
        
        # Retrieve context
        context = manager.get_context(session.session_id)
        
        # Verify context is valid
        assert context is not None
        assert len(context["conversation_history"]) == len(turns)
        assert context["turn_count"] == len(turns)
        assert context["is_active"] is True
