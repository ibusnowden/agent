"""
Unit Tests for Judge System

Tests all judge types and verdict mechanisms.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.judge import (
    Judge,
    LLMJudge,
    GroundedJudge,
    CodeSecurityJudge,
    ConsensusJudge,
    MetaJudge,
    Verdict,
    JudgmentResult,
)
from core.llm_provider import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM for judge testing."""

    def __init__(self, verdict_to_return="AGENT_A"):
        super().__init__("mock-model")
        self.verdict_to_return = verdict_to_return

    def generate(self, prompt, system_prompt=None, **kwargs):
        # Return verdict based on configuration
        if self.verdict_to_return == "AGENT_A":
            content = "Winner: Agent A\nConfidence: 85\nReasoning: Agent A provided better answer."
        elif self.verdict_to_return == "AGENT_B":
            content = "Winner: Agent B\nConfidence: 90\nReasoning: Agent B was more accurate."
        elif self.verdict_to_return == "TIE":
            content = "Winner: TIE\nConfidence: 75\nReasoning: Both answers equally good."
        else:  # PASS/FAIL for single evaluation
            content = f"Verdict: {self.verdict_to_return}\nConfidence: 80\nReasoning: Test reasoning."

        return LLMResponse(
            content=content,
            model="mock",
            provider="mock",
            tokens_used=10
        )

    def generate_batch(self, prompts, **kwargs):
        return [self.generate(p, **kwargs) for p in prompts]


class TestVerdict:
    """Test Verdict enum."""

    def test_verdict_values(self):
        """Test verdict enum values."""
        assert Verdict.AGENT_A_WINS.value == "agent_a"
        assert Verdict.AGENT_B_WINS.value == "agent_b"
        assert Verdict.TIE.value == "tie"
        assert Verdict.INVALID.value == "invalid"


class TestJudgmentResult:
    """Test JudgmentResult dataclass."""

    def test_creation(self):
        """Test creating judgment result."""
        result = JudgmentResult(
            verdict=Verdict.AGENT_A_WINS,
            reasoning="Test reasoning",
            confidence=0.85
        )

        assert result.verdict == Verdict.AGENT_A_WINS
        assert result.reasoning == "Test reasoning"
        assert result.confidence == 0.85
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test judgment with metadata."""
        result = JudgmentResult(
            verdict=Verdict.TIE,
            reasoning="Equal quality",
            confidence=0.5,
            metadata={"task": "test"}
        )

        assert result.metadata["task"] == "test"


class TestLLMJudge:
    """Test LLM-based judge."""

    def test_initialization(self):
        """Test LLM judge initialization."""
        llm = MockLLMProvider()
        judge = LLMJudge("judge_1", llm)

        assert judge.judge_id == "judge_1"
        assert judge.llm == llm
        assert len(judge.judgment_history) == 0

    def test_single_response_evaluation(self):
        """Test judging a single response."""
        llm = MockLLMProvider(verdict_to_return="PASS")
        judge = LLMJudge("judge_1", llm)

        context = {
            "task": "What is 2+2?",
            "response_a": "2+2 equals 4"
        }

        result = judge.judge(context)

        assert isinstance(result, JudgmentResult)
        assert result.verdict == Verdict.AGENT_A_WINS  # PASS maps to AGENT_A_WINS
        assert 0.0 <= result.confidence <= 1.0
        assert len(judge.judgment_history) == 1

    def test_comparative_evaluation(self):
        """Test comparing two responses."""
        llm = MockLLMProvider(verdict_to_return="AGENT_A")
        judge = LLMJudge("judge_1", llm)

        context = {
            "task": "What is 2+2?",
            "response_a": "2+2 equals 4",
            "response_b": "2+2 equals 5"
        }

        result = judge.judge(context)

        assert isinstance(result, JudgmentResult)
        assert result.verdict == Verdict.AGENT_A_WINS
        assert len(judge.judgment_history) == 1

    def test_tie_verdict(self):
        """Test tie verdict."""
        llm = MockLLMProvider(verdict_to_return="TIE")
        judge = LLMJudge("judge_1", llm)

        context = {
            "task": "Test",
            "response_a": "Response A",
            "response_b": "Response B"
        }

        result = judge.judge(context)
        assert result.verdict == Verdict.TIE

    def test_judgment_history(self):
        """Test that judgments are recorded."""
        llm = MockLLMProvider()
        judge = LLMJudge("judge_1", llm)

        # Make multiple judgments
        for i in range(3):
            context = {"task": f"Task {i}", "response_a": f"Response {i}"}
            judge.judge(context)

        assert len(judge.judgment_history) == 3


class TestGroundedJudge:
    """Test grounded (objective) judge."""

    def test_successful_verification(self):
        """Test verification that passes."""
        def verify_fn(context):
            return True, "Verification passed"

        judge = GroundedJudge("grounded_1", verify_fn)
        result = judge.judge({"test": "data"})

        assert result.verdict == Verdict.AGENT_A_WINS
        assert result.confidence == 1.0
        assert "passed" in result.reasoning.lower()

    def test_failed_verification(self):
        """Test verification that fails."""
        def verify_fn(context):
            return False, "Verification failed"

        judge = GroundedJudge("grounded_1", verify_fn)
        result = judge.judge({"test": "data"})

        assert result.verdict == Verdict.INVALID
        assert result.confidence == 1.0
        assert "failed" in result.reasoning.lower()

    def test_verification_error(self):
        """Test handling verification errors."""
        def verify_fn(context):
            raise ValueError("Test error")

        judge = GroundedJudge("grounded_1", verify_fn)
        result = judge.judge({"test": "data"})

        assert result.verdict == Verdict.INVALID
        assert result.confidence == 0.0
        assert "error" in result.reasoning.lower()


class TestCodeSecurityJudge:
    """Test code security judge."""

    def test_detects_eval_exec(self):
        """Test detection of eval/exec."""
        judge = CodeSecurityJudge("security_1")

        context = {"code": "eval(user_input)"}
        result = judge.judge(context)

        assert result.verdict == Verdict.INVALID
        assert "eval" in result.reasoning.lower()

    def test_detects_sql_injection(self):
        """Test detection of SQL injection."""
        judge = CodeSecurityJudge("security_1")

        context = {
            "code": 'query = "SELECT * FROM users WHERE name = \'" + username + "\'\"'
        }
        result = judge.judge(context)

        assert result.verdict == Verdict.INVALID
        assert "sql" in result.reasoning.lower()

    def test_safe_code_passes(self):
        """Test that safe code passes."""
        judge = CodeSecurityJudge("security_1")

        context = {
            "code": """
def safe_function(x):
    return x * 2
            """
        }
        result = judge.judge(context)

        assert result.verdict == Verdict.AGENT_A_WINS


class TestConsensusJudge:
    """Test consensus judge."""

    def test_majority_voting(self):
        """Test majority voting strategy."""
        # Create 3 judges with different verdicts
        llm1 = MockLLMProvider(verdict_to_return="AGENT_A")
        llm2 = MockLLMProvider(verdict_to_return="AGENT_A")
        llm3 = MockLLMProvider(verdict_to_return="AGENT_B")

        judge1 = LLMJudge("j1", llm1)
        judge2 = LLMJudge("j2", llm2)
        judge3 = LLMJudge("j3", llm3)

        consensus = ConsensusJudge(
            "consensus_1",
            [judge1, judge2, judge3],
            voting_strategy="majority"
        )

        context = {
            "task": "Test",
            "response_a": "A",
            "response_b": "B"
        }

        result = consensus.judge(context)

        # Should choose AGENT_A (2 votes vs 1)
        assert result.verdict == Verdict.AGENT_A_WINS
        assert result.metadata["num_judges"] == 3

    def test_weighted_voting(self):
        """Test weighted voting by confidence."""
        # Would need to mock confidences differently for full test
        # For now, just test that it runs
        llm = MockLLMProvider()
        judge1 = LLMJudge("j1", llm)
        judge2 = LLMJudge("j2", llm)

        consensus = ConsensusJudge(
            "consensus_1",
            [judge1, judge2],
            voting_strategy="weighted"
        )

        context = {"task": "Test", "response_a": "A"}
        result = consensus.judge(context)

        assert isinstance(result, JudgmentResult)

    def test_handles_judge_failures(self):
        """Test handling when some judges fail."""
        def failing_verify(context):
            raise Exception("Judge failed")

        judge1 = GroundedJudge("j1", failing_verify)
        llm = MockLLMProvider()
        judge2 = LLMJudge("j2", llm)

        consensus = ConsensusJudge("consensus_1", [judge1, judge2])

        context = {"task": "Test", "response_a": "A"}
        result = consensus.judge(context)

        # Should still work with one judge
        assert isinstance(result, JudgmentResult)


class TestMetaJudge:
    """Test meta-learning judge."""

    def test_initialization(self):
        """Test meta judge initialization."""
        llm = MockLLMProvider()
        judge = MetaJudge("meta_1", llm)

        assert len(judge.feedback_history) == 0

    def test_receive_feedback(self):
        """Test receiving feedback."""
        llm = MockLLMProvider(verdict_to_return="AGENT_A")
        judge = MetaJudge("meta_1", llm)

        # Make a judgment
        context = {"task": "Test", "response_a": "A", "response_b": "B"}
        judge.judge(context)

        # Give feedback
        judge.receive_feedback(0, Verdict.AGENT_A_WINS, "Correct!")

        assert len(judge.feedback_history) == 1
        assert judge.feedback_history[0]["was_correct"] is True

    def test_accuracy_tracking(self):
        """Test accuracy computation."""
        llm = MockLLMProvider(verdict_to_return="AGENT_A")
        judge = MetaJudge("meta_1", llm)

        # Make judgments
        for i in range(5):
            context = {"task": f"Test {i}", "response_a": "A", "response_b": "B"}
            judge.judge(context)

        # Give feedback (3 correct, 2 incorrect)
        judge.receive_feedback(0, Verdict.AGENT_A_WINS, "Correct")  # correct
        judge.receive_feedback(1, Verdict.AGENT_A_WINS, "Correct")  # correct
        judge.receive_feedback(2, Verdict.AGENT_B_WINS, "Wrong")    # incorrect
        judge.receive_feedback(3, Verdict.AGENT_A_WINS, "Correct")  # correct
        judge.receive_feedback(4, Verdict.AGENT_B_WINS, "Wrong")    # incorrect

        accuracy = judge._compute_accuracy()
        assert accuracy == 0.6  # 3/5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
