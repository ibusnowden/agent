"""
Unit Tests for State Machines

Tests co-evolution and debate state machines.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state_machine import (
    CoEvolutionStateMachine,
    DebateStateMachine,
    CoEvolutionState,
    CoEvolutionResult,
)
from core.agent import ActorAgent, SupervisorAgent
from core.judge import LLMJudge, Verdict, JudgmentResult
from core.llm_provider import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM for state machine testing."""

    def __init__(self, responses=None):
        super().__init__("mock-model")
        self.responses = responses or []
        self.call_count = 0

    def generate(self, prompt, system_prompt=None, **kwargs):
        self.call_count += 1

        # Return predefined responses or default
        if self.responses and self.call_count <= len(self.responses):
            content = self.responses[self.call_count - 1]
        else:
            content = f"Mock response #{self.call_count}"

        return LLMResponse(
            content=content,
            model="mock",
            provider="mock",
            tokens_used=10
        )

    def generate_batch(self, prompts, **kwargs):
        return [self.generate(p, **kwargs) for p in prompts]


class MockJudge:
    """Mock judge for testing."""

    def __init__(self, verdicts_to_return=None):
        self.verdicts = verdicts_to_return or [Verdict.AGENT_A_WINS]
        self.call_count = 0
        self.judge_id = "mock_judge"
        self.judgment_history = []

    def judge(self, context):
        verdict = self.verdicts[min(self.call_count, len(self.verdicts) - 1)]
        self.call_count += 1

        # Low confidence for first calls, high for last
        confidence = 0.5 if self.call_count < len(self.verdicts) else 0.9

        result = JudgmentResult(
            verdict=verdict,
            reasoning="Mock judgment",
            confidence=confidence
        )

        self.judgment_history.append(result)
        return result


class TestCoEvolutionStateMachine:
    """Test co-evolution state machine."""

    def test_initialization(self):
        """Test state machine initialization."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor", llm)
        supervisor = SupervisorAgent("supervisor", llm)
        judge = MockJudge()

        machine = CoEvolutionStateMachine(
            actor=actor,
            supervisor=supervisor,
            judge=judge,
            max_iterations=3
        )

        assert machine.actor == actor
        assert machine.supervisor == supervisor
        assert machine.judge == judge
        assert machine.max_iterations == 3

    def test_single_iteration_success(self):
        """Test successful completion in one iteration."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor", llm)
        supervisor = SupervisorAgent("supervisor", llm)

        # Judge approves immediately (high confidence)
        judge = MockJudge(verdicts_to_return=[Verdict.AGENT_A_WINS])

        machine = CoEvolutionStateMachine(actor, supervisor, judge, max_iterations=5)

        result = machine.run("Test task")

        assert isinstance(result, CoEvolutionResult)
        assert result.iterations >= 1
        assert result.judgment.verdict == Verdict.AGENT_A_WINS
        assert len(result.history) > 0

    def test_multiple_iterations(self):
        """Test iteration until success."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor", llm)
        supervisor = SupervisorAgent("supervisor", llm)

        # Judge rejects twice, then approves
        judge = MockJudge(verdicts_to_return=[
            Verdict.INVALID,
            Verdict.INVALID,
            Verdict.AGENT_A_WINS
        ])

        machine = CoEvolutionStateMachine(actor, supervisor, judge, max_iterations=5)

        result = machine.run("Test task")

        # Should iterate multiple times
        assert result.iterations == 3
        assert judge.call_count == 3

    def test_max_iterations_reached(self):
        """Test termination at max iterations."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor", llm)
        supervisor = SupervisorAgent("supervisor", llm)

        # Judge always rejects (low confidence)
        judge = MockJudge(verdicts_to_return=[Verdict.INVALID] * 10)

        machine = CoEvolutionStateMachine(actor, supervisor, judge, max_iterations=3)

        result = machine.run("Test task")

        # Should stop at max iterations
        assert result.iterations == 3

    def test_history_tracking(self):
        """Test that history is tracked correctly."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor", llm)
        supervisor = SupervisorAgent("supervisor", llm)
        judge = MockJudge()

        machine = CoEvolutionStateMachine(actor, supervisor, judge, max_iterations=2)

        result = machine.run("Test task")

        # History should contain entries from actor, supervisor, judge
        assert len(result.history) > 0
        steps = [entry["step"] for entry in result.history]
        assert "actor" in steps
        assert "supervisor" in steps
        assert "judge" in steps

    def test_token_tracking(self):
        """Test token usage tracking."""
        llm = MockLLMProvider()
        actor = ActorAgent("actor", llm)
        supervisor = SupervisorAgent("supervisor", llm)
        judge = MockJudge()

        machine = CoEvolutionStateMachine(actor, supervisor, judge, max_iterations=2)

        result = machine.run("Test task")

        # Should have accumulated tokens
        assert result.total_tokens > 0
        assert result.total_tokens == actor.total_tokens_used + supervisor.total_tokens_used


class TestDebateStateMachine:
    """Test debate state machine."""

    def test_initialization(self):
        """Test debate machine initialization."""
        llm = MockLLMProvider()
        actor_a = ActorAgent("actor_a", llm)
        actor_b = ActorAgent("actor_b", llm)
        judge = MockJudge()

        debate = DebateStateMachine(
            actor_a=actor_a,
            actor_b=actor_b,
            judge=judge,
            max_turns=3
        )

        assert debate.actor_a == actor_a
        assert debate.actor_b == actor_b
        assert debate.judge == judge
        assert debate.max_turns == 3

    def test_debate_execution(self):
        """Test running a debate."""
        llm_a = MockLLMProvider(responses=["Argument A1", "Argument A2", "Argument A3"])
        llm_b = MockLLMProvider(responses=["Argument B1", "Argument B2", "Argument B3"])

        actor_a = ActorAgent("actor_a", llm_a)
        actor_b = ActorAgent("actor_b", llm_b)
        judge = MockJudge(verdicts_to_return=[Verdict.AGENT_A_WINS])

        debate = DebateStateMachine(actor_a, actor_b, judge, max_turns=3)

        result = debate.run("Should AI be regulated?")

        assert isinstance(result, CoEvolutionResult)
        assert result.iterations == 3  # 3 turns
        assert len(result.history) == 6  # 3 turns × 2 agents

    def test_debate_history(self):
        """Test debate history contains both agents."""
        llm = MockLLMProvider()
        actor_a = ActorAgent("actor_a", llm)
        actor_b = ActorAgent("actor_b", llm)
        judge = MockJudge()

        debate = DebateStateMachine(actor_a, actor_b, judge, max_turns=2)

        result = debate.run("Test topic")

        # Check that both agents appear in history
        agents = [entry["agent"] for entry in result.history]
        assert "actor_a" in agents
        assert "actor_b" in agents

        # Check turn numbers
        turns = [entry["turn"] for entry in result.history]
        assert max(turns) == 2

    def test_judge_receives_both_arguments(self):
        """Test that judge sees arguments from both sides."""
        llm = MockLLMProvider()
        actor_a = ActorAgent("actor_a", llm)
        actor_b = ActorAgent("actor_b", llm)

        # Custom judge to inspect context
        class InspectingJudge:
            def __init__(self):
                self.judge_id = "inspect"
                self.last_context = None
                self.judgment_history = []

            def judge(self, context):
                self.last_context = context
                result = JudgmentResult(
                    verdict=Verdict.TIE,
                    reasoning="Test",
                    confidence=0.8
                )
                self.judgment_history.append(result)
                return result

        judge = InspectingJudge()

        debate = DebateStateMachine(actor_a, actor_b, judge, max_turns=2)
        result = debate.run("Test topic")

        # Judge should have seen arguments from both agents
        assert judge.last_context is not None
        assert "response_a" in judge.last_context
        assert "response_b" in judge.last_context


class TestCoEvolutionState:
    """Test the state TypedDict."""

    def test_state_structure(self):
        """Test that state has expected structure."""
        # This is mainly for documentation
        state: CoEvolutionState = {
            "task": "test",
            "task_type": "general",
            "actor_response": None,
            "supervisor_critique": None,
            "judgment": None,
            "iteration_count": 0,
            "history": [],
            "continue_loop": True,
            "max_iterations": 5
        }

        assert state["task"] == "test"
        assert state["iteration_count"] == 0
        assert isinstance(state["history"], list)


class TestCoEvolutionResult:
    """Test the result dataclass."""

    def test_result_creation(self):
        """Test creating result object."""
        judgment = JudgmentResult(
            verdict=Verdict.AGENT_A_WINS,
            reasoning="Good answer",
            confidence=0.9
        )

        result = CoEvolutionResult(
            final_response="Final answer",
            iterations=3,
            judgment=judgment,
            history=[{"step": 1}],
            total_tokens=100
        )

        assert result.final_response == "Final answer"
        assert result.iterations == 3
        assert result.judgment == judgment
        assert result.total_tokens == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
