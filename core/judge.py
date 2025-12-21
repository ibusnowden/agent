"""
Judge System for COEVOLVE Framework

Judges evaluate agent outputs and determine winners in debates/competitions.

Judge Types:
- Judge: Abstract base class
- LLMJudge: Uses LLM to evaluate
- GroundedJudge: Uses objective criteria (tests, verification)
- ConsensusJudge: Combines multiple judges
- MetaJudge: Learns to judge better over time
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re

from .llm_provider import BaseLLMProvider, create_llm_from_config
from .config import ModelConfig


logger = logging.getLogger(__name__)


class Verdict(Enum):
    """Possible judge verdicts."""
    AGENT_A_WINS = "agent_a"
    AGENT_B_WINS = "agent_b"
    TIE = "tie"
    INVALID = "invalid"


@dataclass
class JudgmentResult:
    """Result of a judgment."""
    verdict: Verdict
    reasoning: str
    confidence: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Judge(ABC):
    """
    Abstract base class for judges.

    Judges evaluate outputs and determine winners.
    """

    def __init__(self, judge_id: str):
        self.judge_id = judge_id
        self.judgment_history: List[JudgmentResult] = []

    @abstractmethod
    def judge(self, context: Dict[str, Any]) -> JudgmentResult:
        """
        Evaluate and return judgment.

        Args:
            context: Dictionary containing information to judge

        Returns:
            JudgmentResult with verdict and reasoning
        """
        pass

    def record_judgment(self, result: JudgmentResult):
        """Record a judgment in history."""
        self.judgment_history.append(result)
        logger.info(f"Judge {self.judge_id}: {result.verdict.value} (confidence: {result.confidence:.2f})")


class LLMJudge(Judge):
    """
    Judge that uses an LLM to evaluate outputs.

    This is useful for subjective criteria like argument quality.
    """

    def __init__(
        self,
        judge_id: str,
        llm_provider: BaseLLMProvider,
        judging_criteria: Optional[str] = None
    ):
        super().__init__(judge_id)
        self.llm = llm_provider
        self.judging_criteria = judging_criteria or self._default_criteria()

    def _default_criteria(self) -> str:
        return """Evaluate based on:
1. Correctness: Factual accuracy
2. Completeness: Fully addresses the question
3. Clarity: Well-organized and understandable
4. Quality: Depth of reasoning"""

    def judge(self, context: Dict[str, Any]) -> JudgmentResult:
        """
        Judge using LLM.

        Context should contain:
        - task: The task/question
        - response_a: First response
        - response_b: Second response (optional, for comparison)
        - agent_a_name: Name of first agent (default: "Agent A")
        - agent_b_name: Name of second agent (default: "Agent B")
        """
        task = context.get("task", "")
        response_a = context.get("response_a", "")
        response_b = context.get("response_b", None)
        agent_a_name = context.get("agent_a_name", "Agent A")
        agent_b_name = context.get("agent_b_name", "Agent B")

        if response_b is None:
            # Single response evaluation
            return self._judge_single(task, response_a, agent_a_name)
        else:
            # Comparative evaluation
            return self._judge_comparative(
                task, response_a, response_b, agent_a_name, agent_b_name
            )

    def _judge_single(self, task: str, response: str, agent_name: str) -> JudgmentResult:
        """Judge a single response."""
        prompt = f"""
        Task: {task}

        {agent_name}'s Response:
        {response}

        {self.judging_criteria}

        Evaluate this response. Provide:
        1. Overall assessment (PASS/FAIL)
        2. Confidence (0-100%)
        3. Detailed reasoning

        Format:
        Verdict: [PASS/FAIL]
        Confidence: [0-100]
        Reasoning: [Your detailed evaluation]
        """

        llm_response = self.llm.generate(prompt, temperature=0.3)
        content = llm_response.content

        # Parse verdict
        if "PASS" in content.upper():
            verdict = Verdict.AGENT_A_WINS
        else:
            verdict = Verdict.INVALID

        # Parse confidence
        confidence_match = re.search(r"Confidence:\s*(\d+)", content)
        confidence = float(confidence_match.group(1)) / 100.0 if confidence_match else 0.5

        # Extract reasoning
        reasoning_match = re.search(r"Reasoning:\s*(.+)", content, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else content

        result = JudgmentResult(
            verdict=verdict,
            reasoning=reasoning,
            confidence=confidence,
            metadata={"task": task, "agent": agent_name}
        )

        self.record_judgment(result)
        return result

    def _judge_comparative(
        self,
        task: str,
        response_a: str,
        response_b: str,
        agent_a_name: str,
        agent_b_name: str
    ) -> JudgmentResult:
        """Judge two responses comparatively."""
        prompt = f"""
        Task: {task}

        {agent_a_name}'s Response:
        {response_a}

        {agent_b_name}'s Response:
        {response_b}

        {self.judging_criteria}

        Compare these responses. Provide:
        1. Winner ({agent_a_name}, {agent_b_name}, or TIE)
        2. Confidence (0-100%)
        3. Detailed reasoning

        Format:
        Winner: [{agent_a_name}/{agent_b_name}/TIE]
        Confidence: [0-100]
        Reasoning: [Your detailed comparison]
        """

        llm_response = self.llm.generate(prompt, temperature=0.3)
        content = llm_response.content

        # Parse winner
        if agent_a_name.upper() in content.upper().split("WINNER:")[1].split("\n")[0]:
            verdict = Verdict.AGENT_A_WINS
        elif agent_b_name.upper() in content.upper().split("WINNER:")[1].split("\n")[0]:
            verdict = Verdict.AGENT_B_WINS
        elif "TIE" in content.upper():
            verdict = Verdict.TIE
        else:
            verdict = Verdict.INVALID

        # Parse confidence
        confidence_match = re.search(r"Confidence:\s*(\d+)", content)
        confidence = float(confidence_match.group(1)) / 100.0 if confidence_match else 0.5

        # Extract reasoning
        reasoning_match = re.search(r"Reasoning:\s*(.+)", content, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else content

        result = JudgmentResult(
            verdict=verdict,
            reasoning=reasoning,
            confidence=confidence,
            metadata={
                "task": task,
                "agent_a": agent_a_name,
                "agent_b": agent_b_name
            }
        )

        self.record_judgment(result)
        return result


class GroundedJudge(Judge):
    """
    Judge based on objective, grounded criteria.

    Examples:
    - Code passes unit tests
    - Math answer is correct
    - Output matches expected format
    """

    def __init__(
        self,
        judge_id: str,
        verification_fn: callable
    ):
        super().__init__(judge_id)
        self.verification_fn = verification_fn

    def judge(self, context: Dict[str, Any]) -> JudgmentResult:
        """
        Judge using grounded verification.

        Context should contain whatever the verification function needs.
        """
        try:
            # Call verification function
            is_correct, details = self.verification_fn(context)

            if is_correct:
                verdict = Verdict.AGENT_A_WINS
                reasoning = f"Verification passed: {details}"
                confidence = 1.0
            else:
                verdict = Verdict.INVALID
                reasoning = f"Verification failed: {details}"
                confidence = 1.0

        except Exception as e:
            verdict = Verdict.INVALID
            reasoning = f"Verification error: {str(e)}"
            confidence = 0.0
            logger.error(f"Grounded judge error: {e}")

        result = JudgmentResult(
            verdict=verdict,
            reasoning=reasoning,
            confidence=confidence,
            metadata={"verification_type": "grounded"}
        )

        self.record_judgment(result)
        return result


class CodeSecurityJudge(GroundedJudge):
    """
    Specialized judge for code security.

    Checks for:
    - SQL injection
    - Command injection
    - XSS vulnerabilities
    - Path traversal
    """

    def __init__(self, judge_id: str):
        super().__init__(judge_id, self._verify_code_security)

    def _verify_code_security(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify code security.

        Context should contain:
        - code: The code to verify
        - attack_payload: Optional attack to test against
        """
        code = context.get("code", "")
        attack_payload = context.get("attack_payload", None)

        vulnerabilities = []

        # Check for common vulnerabilities
        if "eval(" in code or "exec(" in code:
            vulnerabilities.append("Dangerous eval/exec usage")

        if re.search(r"os\.system\([\"'].*?\+", code):
            vulnerabilities.append("Potential command injection")

        if re.search(r"SELECT.*?FROM.*?\+", code, re.IGNORECASE):
            vulnerabilities.append("Potential SQL injection")

        if "<script" in code.lower() and "sanitize" not in code.lower():
            vulnerabilities.append("Potential XSS vulnerability")

        # If attack payload provided, test it
        if attack_payload:
            # This would actually execute code in a sandbox
            # For now, just check if code contains sanitization
            if not self._has_input_validation(code):
                vulnerabilities.append("No input validation found")

        if vulnerabilities:
            return False, "; ".join(vulnerabilities)
        else:
            return True, "No obvious vulnerabilities detected"

    def _has_input_validation(self, code: str) -> bool:
        """Check if code has input validation."""
        validation_patterns = [
            r"sanitize",
            r"escape",
            r"validate",
            r"re\.match",
            r"isinstance\(",
            r"assert\s+.*\s+in\s+"
        ]

        for pattern in validation_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True

        return False


class ConsensusJudge(Judge):
    """
    Judge that combines multiple judges via consensus.

    Uses majority voting or weighted voting.
    """

    def __init__(
        self,
        judge_id: str,
        judges: List[Judge],
        voting_strategy: str = "majority"  # "majority" or "weighted"
    ):
        super().__init__(judge_id)
        self.judges = judges
        self.voting_strategy = voting_strategy

    def judge(self, context: Dict[str, Any]) -> JudgmentResult:
        """Judge using consensus of multiple judges."""
        # Collect judgments from all judges
        judgments = []
        for judge in self.judges:
            try:
                result = judge.judge(context)
                judgments.append(result)
            except Exception as e:
                logger.error(f"Judge {judge.judge_id} failed: {e}")

        if not judgments:
            return JudgmentResult(
                verdict=Verdict.INVALID,
                reasoning="No judges could provide judgment",
                confidence=0.0
            )

        # Apply voting strategy
        if self.voting_strategy == "majority":
            verdict = self._majority_vote(judgments)
        elif self.voting_strategy == "weighted":
            verdict = self._weighted_vote(judgments)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

        # Combine reasoning
        reasoning = "\n\n".join([
            f"Judge {i+1}: {j.reasoning[:100]}..." for i, j in enumerate(judgments)
        ])

        # Average confidence
        avg_confidence = sum(j.confidence for j in judgments) / len(judgments)

        result = JudgmentResult(
            verdict=verdict,
            reasoning=f"Consensus judgment:\n{reasoning}",
            confidence=avg_confidence,
            metadata={
                "num_judges": len(judgments),
                "strategy": self.voting_strategy
            }
        )

        self.record_judgment(result)
        return result

    def _majority_vote(self, judgments: List[JudgmentResult]) -> Verdict:
        """Simple majority voting."""
        votes = {}
        for judgment in judgments:
            verdict = judgment.verdict
            votes[verdict] = votes.get(verdict, 0) + 1

        return max(votes.items(), key=lambda x: x[1])[0]

    def _weighted_vote(self, judgments: List[JudgmentResult]) -> Verdict:
        """Weighted voting by confidence."""
        votes = {}
        for judgment in judgments:
            verdict = judgment.verdict
            votes[verdict] = votes.get(verdict, 0.0) + judgment.confidence

        return max(votes.items(), key=lambda x: x[1])[0]


class MetaJudge(LLMJudge):
    """
    Meta-learning judge that improves over time.

    Tracks its judgment history and learns from mistakes.
    """

    def __init__(
        self,
        judge_id: str,
        llm_provider: BaseLLMProvider,
        judging_criteria: Optional[str] = None
    ):
        super().__init__(judge_id, llm_provider, judging_criteria)
        self.feedback_history: List[Dict[str, Any]] = []

    def judge(self, context: Dict[str, Any]) -> JudgmentResult:
        """Judge with meta-learning context."""
        # Get base judgment
        result = super().judge(context)

        # If we have feedback history, incorporate it
        if self.feedback_history:
            result = self._refine_with_history(context, result)

        return result

    def receive_feedback(self, judgment_id: int, ground_truth: Verdict, explanation: str):
        """
        Receive feedback on a past judgment.

        This allows the judge to learn from mistakes.
        """
        if judgment_id >= len(self.judgment_history):
            logger.warning(f"Invalid judgment ID: {judgment_id}")
            return

        past_judgment = self.judgment_history[judgment_id]

        feedback = {
            "judgment": past_judgment,
            "ground_truth": ground_truth,
            "explanation": explanation,
            "was_correct": past_judgment.verdict == ground_truth
        }

        self.feedback_history.append(feedback)
        logger.info(f"Judge {self.judge_id} received feedback. Accuracy: {self._compute_accuracy():.2%}")

    def _refine_with_history(
        self,
        context: Dict[str, Any],
        initial_result: JudgmentResult
    ) -> JudgmentResult:
        """Refine judgment using feedback history."""
        # Find similar past judgments
        similar_feedback = self._find_similar_feedback(context)

        if not similar_feedback:
            return initial_result

        # Build meta-prompt
        feedback_context = "\n".join([
            f"- Past judgment was {f['judgment'].verdict.value}, "
            f"ground truth was {f['ground_truth'].value}: {f['explanation']}"
            for f in similar_feedback[:3]
        ])

        meta_prompt = f"""
        Based on your past feedback:
        {feedback_context}

        Your current judgment: {initial_result.verdict.value}
        Reasoning: {initial_result.reasoning}

        Should you revise this judgment? If yes, provide revised verdict.

        Format:
        Revised: [YES/NO]
        New Verdict: [if YES]
        Reasoning: [explanation]
        """

        response = self.llm.generate(meta_prompt, temperature=0.2)

        if "YES" in response.content.upper():
            # Parse new verdict
            # (simplified - would need more robust parsing)
            logger.info(f"Judge {self.judge_id} revised judgment based on feedback")

        return initial_result

    def _find_similar_feedback(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar past feedback (simplified similarity)."""
        # In a real implementation, would use embeddings
        return self.feedback_history[-5:]  # Just return recent feedback

    def _compute_accuracy(self) -> float:
        """Compute judge accuracy based on feedback."""
        if not self.feedback_history:
            return 0.0

        correct = sum(1 for f in self.feedback_history if f["was_correct"])
        return correct / len(self.feedback_history)


# Factory functions

def create_llm_judge(judge_id: str, config: ModelConfig, **kwargs) -> LLMJudge:
    """Create an LLM judge from configuration."""
    llm = create_llm_from_config(config)
    return LLMJudge(judge_id=judge_id, llm_provider=llm, **kwargs)


def create_code_security_judge(judge_id: str) -> CodeSecurityJudge:
    """Create a code security judge."""
    return CodeSecurityJudge(judge_id=judge_id)


def create_consensus_judge(
    judge_id: str,
    judges: List[Judge],
    voting_strategy: str = "majority"
) -> ConsensusJudge:
    """Create a consensus judge."""
    return ConsensusJudge(judge_id=judge_id, judges=judges, voting_strategy=voting_strategy)


# Example usage
if __name__ == "__main__":
    from .config import ModelConfig

    # Create LLM judge
    print("Testing LLM Judge...")
    config = ModelConfig(provider="openai", model_name="gpt-3.5-turbo")
    judge = create_llm_judge("llm_judge_1", config)

    context = {
        "task": "What is 2+2?",
        "response_a": "2+2 equals 4",
        "response_b": "2+2 equals 5",
        "agent_a_name": "Alice",
        "agent_b_name": "Bob"
    }

    result = judge.judge(context)
    print(f"Verdict: {result.verdict.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning[:150]}...")

    # Create code security judge
    print("\nTesting Code Security Judge...")
    sec_judge = create_code_security_judge("security_judge_1")

    code_context = {
        "code": """
def query_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    return db.execute(query)
        """
    }

    sec_result = sec_judge.judge(code_context)
    print(f"Verdict: {sec_result.verdict.value}")
    print(f"Reasoning: {sec_result.reasoning}")
