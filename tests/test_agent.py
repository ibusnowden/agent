"""
Unit Tests for Agent System

Tests all agent types and their behaviors.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import (
    BaseAgent,
    ActorAgent,
    SupervisorAgent,
    ConstitutionalAgent,
    AgentAction,
    AgentMemory,
)
from core.llm_provider import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__("mock-model", *args, **kwargs)
        self.call_count = 0
        self.last_prompt = None

    def generate(self, prompt, system_prompt=None, **kwargs):
        self.call_count += 1
        self.last_prompt = prompt

        # Generate different responses based on content
        if "critique" in prompt.lower():
            content = "This is a critique: The response could be improved."
        elif "revise" in prompt.lower():
            content = "This is a revised response with improvements."
        elif "reflection" in prompt.lower():
            content = "Reflection: I learned something new."
        else:
            content = f"Response to prompt (call #{self.call_count})"

        return LLMResponse(
            content=content,
            model="mock-model",
            provider="mock",
            tokens_used=10
        )

    def generate_batch(self, prompts, system_prompt=None, **kwargs):
        return [self.generate(p, system_prompt, **kwargs) for p in prompts]


class TestAgentMemory:
    """Test AgentMemory class."""

    def test_initialization(self):
        """Test memory initialization."""
        memory = AgentMemory()
        assert len(memory.observations) == 0
        assert len(memory.actions) == 0
        assert len(memory.reflections) == 0

    def test_add_observation(self):
        """Test adding observations."""
        memory = AgentMemory()
        memory.add_observation("Test observation")

        assert len(memory.observations) == 1
        assert memory.observations[0] == "Test observation"

    def test_add_action(self):
        """Test adding actions."""
        memory = AgentMemory()
        action = AgentAction(
            agent_id="test",
            action_type="generate",
            content="Test content"
        )
        memory.add_action(action)

        assert len(memory.actions) == 1
        assert memory.actions[0] == action

    def test_get_recent_context(self):
        """Test getting recent context."""
        memory = AgentMemory()
        memory.add_observation("Obs 1")
        memory.add_observation("Obs 2")

        action = AgentAction("test", "generate", "Action 1")
        memory.add_action(action)

        context = memory.get_recent_context(n=5)
        assert "Obs 1" in context
        assert "Obs 2" in context
        assert "Action 1" in context


class TestActorAgent:
    """Test ActorAgent class."""

    def test_initialization(self):
        """Test actor initialization."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor_1", llm)

        assert actor.agent_id == "actor_1"
        assert actor.llm == llm
        assert actor.total_tokens_used == 0

    def test_act_without_feedback(self):
        """Test generating response without feedback."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor_1", llm)

        context = {"task": "Explain RL"}
        action = actor.act(context)

        assert isinstance(action, AgentAction)
        assert action.agent_id == "actor_1"
        assert action.action_type == "generate"
        assert len(action.content) > 0
        assert llm.call_count == 1
        assert actor.total_tokens_used > 0

    def test_act_with_feedback(self):
        """Test generating response with feedback."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor_1", llm)

        context = {
            "task": "Explain RL",
            "previous_feedback": "Be more concise"
        }
        action = actor.act(context)

        assert isinstance(action, AgentAction)
        assert "feedback" in llm.last_prompt.lower()
        assert action.metadata["has_feedback"] is True

    def test_revise(self):
        """Test revising response."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor_1", llm)

        original = "Original response"
        critique = "Needs improvement"

        revised_action = actor.revise(original, critique)

        assert isinstance(revised_action, AgentAction)
        assert revised_action.action_type == "revise"
        assert "revise" in llm.last_prompt.lower()

    def test_observe(self):
        """Test adding observation."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor_1", llm)

        actor.observe("Test observation")
        assert len(actor.memory.observations) == 1

    def test_reflect(self):
        """Test reflection."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor_1", llm)

        # Add some experiences
        actor.observe("Observation 1")
        actor.act({"task": "Test task"})

        reflection = actor.reflect()

        assert isinstance(reflection, str)
        assert len(reflection) > 0
        assert len(actor.memory.reflections) == 1


class TestSupervisorAgent:
    """Test SupervisorAgent class."""

    def test_initialization(self):
        """Test supervisor initialization."""
        llm = MockLLMProvider()
        supervisor = SupervisorAgent("supervisor_1", llm)

        assert supervisor.agent_id == "supervisor_1"
        assert len(supervisor.critique_criteria) > 0

    def test_custom_criteria(self):
        """Test supervisor with custom criteria."""
        llm = MockLLMProvider()
        criteria = ["Criterion 1", "Criterion 2"]
        supervisor = SupervisorAgent("supervisor_1", llm, critique_criteria=criteria)

        assert supervisor.critique_criteria == criteria

    def test_critique(self):
        """Test generating critique."""
        llm = MockLLMProvider()
        supervisor = SupervisorAgent("supervisor_1", llm)

        context = {
            "task": "Explain RL",
            "response": "RL is about learning from rewards"
        }
        action = supervisor.act(context)

        assert isinstance(action, AgentAction)
        assert action.action_type == "critique"
        assert "critique" in llm.last_prompt.lower()

    def test_validate(self):
        """Test validation."""
        llm = MockLLMProvider()
        supervisor = SupervisorAgent("supervisor_1", llm)

        # Mock response to include "VALID"
        def mock_generate(prompt, **kwargs):
            return LLMResponse(
                content="Verdict: VALID\nReason: Good response",
                model="mock",
                provider="mock",
                tokens_used=10
            )

        llm.generate = mock_generate

        is_valid, reason = supervisor.validate("Test task", "Test response")

        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)


class TestConstitutionalAgent:
    """Test ConstitutionalAgent class."""

    def test_initialization(self):
        """Test constitutional agent initialization."""
        llm = MockLLMProvider()
        constitution = ["Be helpful", "Be harmless"]
        agent = ConstitutionalAgent("const_1", llm, constitution)

        assert agent.agent_id == "const_1"
        assert agent.constitution == constitution

    def test_generate_with_self_critique(self):
        """Test generation with self-critique."""
        llm = MockLLMProvider()
        constitution = ["Be clear", "Be accurate"]
        agent = ConstitutionalAgent("const_1", llm, constitution)

        # Mock to return "satisfactory" on second call
        call_count = [0]

        def mock_generate(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call (critique)
                content = "The response is satisfactory and meets all principles."
            else:
                content = "Standard response"

            return LLMResponse(
                content=content,
                model="mock",
                provider="mock",
                tokens_used=10
            )

        llm.generate = mock_generate

        task = "Explain something"
        action = agent.generate_with_self_critique(task)

        assert isinstance(action, AgentAction)
        assert action.action_type == "constitutional_generate"
        # Should have called multiple times (generation + critique)
        assert call_count[0] >= 2


class TestAgentAction:
    """Test AgentAction dataclass."""

    def test_creation(self):
        """Test creating AgentAction."""
        action = AgentAction(
            agent_id="test_agent",
            action_type="generate",
            content="Test content",
            metadata={"key": "value"},
            tokens_used=42
        )

        assert action.agent_id == "test_agent"
        assert action.action_type == "generate"
        assert action.content == "Test content"
        assert action.metadata["key"] == "value"
        assert action.tokens_used == 42


class TestTokenTracking:
    """Test token usage tracking."""

    def test_token_accumulation(self):
        """Test that tokens accumulate correctly."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor_1", llm)

        # Perform multiple actions
        actor.act({"task": "Task 1"})
        actor.act({"task": "Task 2"})
        actor.reflect()

        # Should have accumulated tokens
        assert actor.total_tokens_used > 0
        assert actor.total_tokens_used == llm.call_count * 10  # 10 tokens per call


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
