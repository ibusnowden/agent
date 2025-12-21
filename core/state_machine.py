"""
State Machine for Co-Evolutionary Loops

Uses LangGraph to orchestrate Actor → Supervisor → Judge workflows.

This is the core of "The Scrutiny Room" where agents interact and evolve.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field
import operator
import logging

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not installed. Run: pip install langgraph")

from .agent import BaseAgent, ActorAgent, SupervisorAgent, AgentAction
from .judge import Judge, Verdict, JudgmentResult


logger = logging.getLogger(__name__)


# State definition
class CoEvolutionState(TypedDict):
    """
    State for the co-evolution workflow.

    This state is passed between nodes in the graph.
    """
    # Task
    task: str
    task_type: str  # "debate", "code_security", "hypothesis", etc.

    # Agents
    actor_response: Optional[str]
    supervisor_critique: Optional[str]

    # Judgment
    judgment: Optional[JudgmentResult]
    iteration_count: int

    # History (accumulates over iterations)
    history: Annotated[List[Dict[str, Any]], operator.add]

    # Control flow
    continue_loop: bool
    max_iterations: int


@dataclass
class CoEvolutionResult:
    """Result of a co-evolution run."""
    final_response: str
    iterations: int
    judgment: JudgmentResult
    history: List[Dict[str, Any]]
    total_tokens: int = 0


class CoEvolutionStateMachine:
    """
    State machine for co-evolutionary learning.

    Workflow:
    1. Actor generates response
    2. Supervisor critiques
    3. Judge evaluates
    4. If not satisfied, loop back to Actor with feedback
    5. Else, terminate

    This implements "The Scrutiny Room" concept.
    """

    def __init__(
        self,
        actor: ActorAgent,
        supervisor: SupervisorAgent,
        judge: Judge,
        max_iterations: int = 5
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph required. Run: pip install langgraph")

        self.actor = actor
        self.supervisor = supervisor
        self.judge = judge
        self.max_iterations = max_iterations

        # Build graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()

        logger.info(f"Initialized CoEvolutionStateMachine with {max_iterations} max iterations")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(CoEvolutionState)

        # Add nodes
        workflow.add_node("actor", self._actor_node)
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("judge", self._judge_node)

        # Set entry point
        workflow.set_entry_point("actor")

        # Add edges
        workflow.add_edge("actor", "supervisor")
        workflow.add_edge("supervisor", "judge")

        # Conditional edge from judge
        workflow.add_conditional_edges(
            "judge",
            self._should_continue,
            {
                "continue": "actor",  # Loop back to actor
                "end": END  # Terminate
            }
        )

        return workflow

    def _actor_node(self, state: CoEvolutionState) -> Dict[str, Any]:
        """Actor generates or refines response."""
        logger.debug(f"Actor node (iteration {state['iteration_count']})")

        # Build context for actor
        context = {
            "task": state["task"],
            "previous_feedback": state.get("supervisor_critique")
        }

        # Generate response
        action = self.actor.act(context)
        response = action.content

        # Update state
        return {
            "actor_response": response,
            "history": [{
                "step": "actor",
                "iteration": state["iteration_count"],
                "content": response[:200] + "..."  # Truncate for history
            }]
        }

    def _supervisor_node(self, state: CoEvolutionState) -> Dict[str, Any]:
        """Supervisor critiques actor's response."""
        logger.debug(f"Supervisor node (iteration {state['iteration_count']})")

        # Build context for supervisor
        context = {
            "task": state["task"],
            "response": state["actor_response"]
        }

        # Generate critique
        action = self.supervisor.act(context)
        critique = action.content

        # Update state
        return {
            "supervisor_critique": critique,
            "history": [{
                "step": "supervisor",
                "iteration": state["iteration_count"],
                "content": critique[:200] + "..."
            }]
        }

    def _judge_node(self, state: CoEvolutionState) -> Dict[str, Any]:
        """Judge evaluates the interaction."""
        logger.debug(f"Judge node (iteration {state['iteration_count']})")

        # Build context for judge
        context = {
            "task": state["task"],
            "response_a": state["actor_response"],
            "critique": state["supervisor_critique"]
        }

        # Get judgment
        judgment = self.judge.judge(context)

        # Increment iteration
        new_iteration = state["iteration_count"] + 1

        # Update state
        return {
            "judgment": judgment,
            "iteration_count": new_iteration,
            "history": [{
                "step": "judge",
                "iteration": state["iteration_count"],
                "verdict": judgment.verdict.value,
                "confidence": judgment.confidence
            }]
        }

    def _should_continue(self, state: CoEvolutionState) -> str:
        """
        Decide whether to continue the loop or terminate.

        Terminates if:
        - Judge approves (high confidence AGENT_A_WINS)
        - Max iterations reached
        """
        judgment = state["judgment"]
        iteration = state["iteration_count"]
        max_iter = state["max_iterations"]

        # Check max iterations
        if iteration >= max_iter:
            logger.info(f"Terminating: Max iterations ({max_iter}) reached")
            return "end"

        # Check judgment
        if judgment.verdict == Verdict.AGENT_A_WINS and judgment.confidence > 0.8:
            logger.info(f"Terminating: High confidence approval ({judgment.confidence:.2f})")
            return "end"

        # Continue loop
        logger.info(f"Continuing: Iteration {iteration}, confidence {judgment.confidence:.2f}")
        return "continue"

    def run(self, task: str, task_type: str = "general") -> CoEvolutionResult:
        """
        Run the co-evolution loop.

        Args:
            task: The task for the actor to solve
            task_type: Type of task (for logging/metrics)

        Returns:
            CoEvolutionResult with final output and history
        """
        logger.info(f"Starting co-evolution for task: {task[:50]}...")

        # Initialize state
        initial_state: CoEvolutionState = {
            "task": task,
            "task_type": task_type,
            "actor_response": None,
            "supervisor_critique": None,
            "judgment": None,
            "iteration_count": 0,
            "history": [],
            "continue_loop": True,
            "max_iterations": self.max_iterations
        }

        # Run graph
        final_state = self.app.invoke(initial_state)

        # Build result
        result = CoEvolutionResult(
            final_response=final_state["actor_response"],
            iterations=final_state["iteration_count"],
            judgment=final_state["judgment"],
            history=final_state["history"],
            total_tokens=self.actor.total_tokens_used + self.supervisor.total_tokens_used
        )

        logger.info(f"Co-evolution completed in {result.iterations} iterations")
        logger.info(f"Final verdict: {result.judgment.verdict.value}")

        return result


class DebateStateMachine:
    """
    Specialized state machine for debates.

    Two actors debate, judge picks winner.
    """

    def __init__(
        self,
        actor_a: ActorAgent,
        actor_b: ActorAgent,
        judge: Judge,
        max_turns: int = 4
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph required")

        self.actor_a = actor_a
        self.actor_b = actor_b
        self.judge = judge
        self.max_turns = max_turns

        logger.info(f"Initialized DebateStateMachine with {max_turns} max turns")

    def run(self, topic: str) -> CoEvolutionResult:
        """
        Run a debate.

        Args:
            topic: The debate topic/question

        Returns:
            CoEvolutionResult with debate history and winner
        """
        logger.info(f"Starting debate: {topic[:50]}...")

        debate_history = []
        turn_count = 0

        # Debate loop
        while turn_count < self.max_turns:
            logger.debug(f"Debate turn {turn_count + 1}/{self.max_turns}")

            # Actor A's turn
            context_a = {
                "task": f"Debate topic: {topic}",
                "previous_feedback": debate_history[-1]["content"] if debate_history else None
            }
            action_a = self.actor_a.act(context_a)
            debate_history.append({
                "turn": turn_count + 1,
                "agent": "actor_a",
                "content": action_a.content
            })

            # Actor B's turn
            context_b = {
                "task": f"Debate topic: {topic}",
                "previous_feedback": action_a.content
            }
            action_b = self.actor_b.act(context_b)
            debate_history.append({
                "turn": turn_count + 1,
                "agent": "actor_b",
                "content": action_b.content
            })

            turn_count += 1

        # Judge decides winner
        judge_context = {
            "task": topic,
            "response_a": "\n\n".join([h["content"] for h in debate_history if h["agent"] == "actor_a"]),
            "response_b": "\n\n".join([h["content"] for h in debate_history if h["agent"] == "actor_b"]),
            "agent_a_name": self.actor_a.agent_id,
            "agent_b_name": self.actor_b.agent_id
        }

        judgment = self.judge.judge(judge_context)

        result = CoEvolutionResult(
            final_response=f"Winner: {judgment.verdict.value}",
            iterations=turn_count,
            judgment=judgment,
            history=debate_history,
            total_tokens=self.actor_a.total_tokens_used + self.actor_b.total_tokens_used
        )

        logger.info(f"Debate completed. Winner: {judgment.verdict.value}")
        return result


# Example usage
if __name__ == "__main__":
    from .config import ModelConfig
    from .agent import create_actor, create_supervisor
    from .judge import create_llm_judge

    # Configuration
    config = ModelConfig(provider="openai", model_name="gpt-3.5-turbo")

    # Create agents
    actor = create_actor("actor_1", config)
    supervisor = create_supervisor("supervisor_1", config)
    judge = create_llm_judge("judge_1", config)

    # Create state machine
    print("Testing Co-Evolution State Machine...")
    state_machine = CoEvolutionStateMachine(
        actor=actor,
        supervisor=supervisor,
        judge=judge,
        max_iterations=3
    )

    # Run
    task = "Explain the concept of reinforcement learning in simple terms."
    result = state_machine.run(task)

    print(f"\nFinal response: {result.final_response[:200]}...")
    print(f"Iterations: {result.iterations}")
    print(f"Verdict: {result.judgment.verdict.value}")
    print(f"Confidence: {result.judgment.confidence:.2f}")
    print(f"Total tokens: {result.total_tokens}")

    print("\nHistory:")
    for entry in result.history:
        print(f"  {entry}")
