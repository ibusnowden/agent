"""
Agent Classes for COEVOLVE Framework

Base agent abstractions and specialized agents for co-evolutionary learning.

Agent Types:
- BaseAgent: Abstract base class
- ActorAgent: Generates solutions/responses
- SupervisorAgent: Critiques and validates
- ConstitutionalAgent: Self-critiques using constitution
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .llm_provider import BaseLLMProvider, LLMResponse, create_llm_from_config
from .config import ModelConfig, ConstitutionalAIConfig


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    agent_id: str
    action_type: str  # "generate", "critique", "revise", etc.
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    tokens_used: int = 0


@dataclass
class AgentMemory:
    """Simple memory storage for an agent."""
    observations: List[str] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)

    def add_observation(self, observation: str):
        """Add an observation."""
        self.observations.append(observation)

    def add_action(self, action: AgentAction):
        """Add an action."""
        self.actions.append(action)

    def add_reflection(self, reflection: str):
        """Add a reflection."""
        self.reflections.append(reflection)

    def get_recent_context(self, n: int = 5) -> str:
        """Get recent context as a formatted string."""
        context_parts = []

        # Recent observations
        if self.observations:
            recent_obs = self.observations[-n:]
            context_parts.append("Recent observations:")
            context_parts.extend(f"- {obs}" for obs in recent_obs)

        # Recent actions
        if self.actions:
            recent_actions = self.actions[-n:]
            context_parts.append("\nRecent actions:")
            context_parts.extend(
                f"- [{a.action_type}] {a.content[:100]}..."
                for a in recent_actions
            )

        return "\n".join(context_parts)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    All agents have:
    - A unique ID
    - An LLM provider
    - Memory
    - Ability to act given a context
    """

    def __init__(
        self,
        agent_id: str,
        llm_provider: BaseLLMProvider,
        system_prompt: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.llm = llm_provider
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.memory = AgentMemory()
        self.total_tokens_used = 0

        logger.info(f"Initialized {self.__class__.__name__} with ID: {agent_id}")

    @abstractmethod
    def _default_system_prompt(self) -> str:
        """Return the default system prompt for this agent type."""
        pass

    @abstractmethod
    def act(self, context: Dict[str, Any]) -> AgentAction:
        """
        Take an action given the current context.

        Args:
            context: Dictionary containing relevant information

        Returns:
            AgentAction with the agent's response
        """
        pass

    def _generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Internal method to generate LLM response."""
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt or self.system_prompt,
            **kwargs
        )

        # Track token usage
        self.total_tokens_used += response.tokens_used

        return response

    def observe(self, observation: str):
        """Add an observation to memory."""
        self.memory.add_observation(observation)

    def reflect(self) -> str:
        """Generate a reflection on recent experiences."""
        context = self.memory.get_recent_context(n=10)

        prompt = f"""
        Based on your recent experiences:

        {context}

        Generate a high-level insight or reflection about what you've learned.
        """

        response = self._generate(prompt)
        reflection = response.content

        self.memory.add_reflection(reflection)
        logger.debug(f"Agent {self.agent_id} reflected: {reflection[:100]}...")

        return reflection


class ActorAgent(BaseAgent):
    """
    Actor Agent: Generates solutions, proposals, or responses.

    This agent is the "generator" in the co-evolutionary system.
    It proposes solutions that will be critiqued by supervisors.
    """

    def _default_system_prompt(self) -> str:
        return """You are an Actor agent in a co-evolutionary learning system.
Your role is to generate solutions, proposals, or responses to tasks.
Be creative, thorough, and aim for high quality.
You will receive feedback from supervisors to improve your outputs."""

    def act(self, context: Dict[str, Any]) -> AgentAction:
        """
        Generate a response to the given task.

        Context should contain:
        - task: The task or question to respond to
        - previous_feedback: (optional) Feedback from previous attempts
        """
        task = context.get("task", "")
        previous_feedback = context.get("previous_feedback", None)

        # Build prompt
        if previous_feedback:
            prompt = f"""
            Task: {task}

            Previous attempt received this feedback:
            {previous_feedback}

            Generate an improved response that addresses the feedback.
            Response:
            """
        else:
            prompt = f"""
            Task: {task}

            Generate a high-quality response.
            Response:
            """

        # Generate response
        llm_response = self._generate(prompt)

        # Create action
        action = AgentAction(
            agent_id=self.agent_id,
            action_type="generate",
            content=llm_response.content,
            metadata={
                "task": task,
                "has_feedback": previous_feedback is not None
            },
            tokens_used=llm_response.tokens_used
        )

        self.memory.add_action(action)
        logger.info(f"Actor {self.agent_id} generated response ({len(action.content)} chars)")

        return action

    def revise(self, original_response: str, critique: str) -> AgentAction:
        """
        Revise a response based on critique.

        Args:
            original_response: The original response
            critique: Critique from supervisor

        Returns:
            AgentAction with revised response
        """
        prompt = f"""
        Original response:
        {original_response}

        Critique:
        {critique}

        Revise the response to address the critique while maintaining what was good.
        Revised response:
        """

        llm_response = self._generate(prompt)

        action = AgentAction(
            agent_id=self.agent_id,
            action_type="revise",
            content=llm_response.content,
            metadata={
                "original_length": len(original_response),
                "revised_length": len(llm_response.content)
            },
            tokens_used=llm_response.tokens_used
        )

        self.memory.add_action(action)
        logger.info(f"Actor {self.agent_id} revised response")

        return action


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent: Critiques and validates outputs from actors.

    This agent is the "discriminator" in the co-evolutionary system.
    It provides feedback to help actors improve.
    """

    def __init__(
        self,
        agent_id: str,
        llm_provider: BaseLLMProvider,
        critique_criteria: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ):
        super().__init__(agent_id, llm_provider, system_prompt)
        self.critique_criteria = critique_criteria or self._default_criteria()

    def _default_system_prompt(self) -> str:
        return """You are a Supervisor agent in a co-evolutionary learning system.
Your role is to critique outputs from actor agents.
Be thorough, constructive, and specific in your feedback.
Point out both strengths and weaknesses.
Your critiques help actors improve their outputs."""

    def _default_criteria(self) -> List[str]:
        """Default criteria for critique."""
        return [
            "Correctness: Is the response factually accurate?",
            "Completeness: Does it fully address the task?",
            "Clarity: Is it well-organized and easy to understand?",
            "Quality: Is it well-reasoned and thorough?"
        ]

    def act(self, context: Dict[str, Any]) -> AgentAction:
        """
        Critique an actor's response.

        Context should contain:
        - task: The original task
        - response: The actor's response to critique
        """
        task = context.get("task", "")
        response = context.get("response", "")

        # Build critique prompt
        criteria_str = "\n".join(f"- {c}" for c in self.critique_criteria)

        prompt = f"""
        Task: {task}

        Response to critique:
        {response}

        Evaluate this response based on the following criteria:
        {criteria_str}

        Provide specific, constructive critique:
        1. What is done well?
        2. What needs improvement?
        3. Specific suggestions for improvement.

        Critique:
        """

        # Generate critique
        llm_response = self._generate(prompt)

        action = AgentAction(
            agent_id=self.agent_id,
            action_type="critique",
            content=llm_response.content,
            metadata={
                "task": task,
                "response_length": len(response),
                "criteria_used": len(self.critique_criteria)
            },
            tokens_used=llm_response.tokens_used
        )

        self.memory.add_action(action)
        logger.info(f"Supervisor {self.agent_id} generated critique")

        return action

    def validate(self, task: str, response: str) -> Tuple[bool, str]:
        """
        Validate if a response meets minimum standards.

        Args:
            task: The task
            response: The response to validate

        Returns:
            (is_valid, reason)
        """
        prompt = f"""
        Task: {task}

        Response:
        {response}

        Does this response adequately address the task? Answer with:
        - VALID if it meets minimum standards
        - INVALID if it has critical flaws

        Format:
        Verdict: [VALID/INVALID]
        Reason: [Brief explanation]
        """

        llm_response = self._generate(prompt, temperature=0.3)  # Lower temp for validation

        content = llm_response.content
        is_valid = "VALID" in content and "INVALID" not in content.split("VALID")[0]

        return is_valid, content


class ConstitutionalAgent(ActorAgent):
    """
    Constitutional Agent: Self-critiques using a constitution.

    Based on Constitutional AI (Anthropic, 2022).
    This agent can critique and revise its own outputs.
    """

    def __init__(
        self,
        agent_id: str,
        llm_provider: BaseLLMProvider,
        constitution: List[str],
        config: Optional[ConstitutionalAIConfig] = None,
        system_prompt: Optional[str] = None
    ):
        super().__init__(agent_id, llm_provider, system_prompt)
        self.constitution = constitution
        self.config = config or ConstitutionalAIConfig()

    def _default_system_prompt(self) -> str:
        return """You are a Constitutional agent.
You generate responses and then critique them according to a set of principles.
You iteratively refine your outputs to better align with these principles."""

    def generate_with_self_critique(self, task: str) -> AgentAction:
        """
        Generate a response with iterative self-critique.

        Args:
            task: The task to respond to

        Returns:
            Final revised response after self-critique
        """
        # Initial generation
        initial_response = self.act({"task": task})
        response = initial_response.content

        # Iterative refinement
        for iteration in range(self.config.num_critique_iterations):
            logger.debug(f"Self-critique iteration {iteration + 1}/{self.config.num_critique_iterations}")

            # Self-critique
            critique = self._self_critique(response)

            # Check if satisfactory
            if "no issues" in critique.lower() or "satisfactory" in critique.lower():
                logger.info(f"Self-critique satisfied after {iteration + 1} iterations")
                break

            # Revise based on self-critique
            revised = self._self_revise(response, critique)
            response = revised

        # Final action
        action = AgentAction(
            agent_id=self.agent_id,
            action_type="constitutional_generate",
            content=response,
            metadata={
                "task": task,
                "iterations": iteration + 1,
                "constitution_size": len(self.constitution)
            }
        )

        self.memory.add_action(action)
        return action

    def _self_critique(self, response: str) -> str:
        """Generate critique based on constitution."""
        constitution_str = "\n".join(f"{i+1}. {p}" for i, p in enumerate(self.constitution))

        prompt = f"""
        Constitution (principles to follow):
        {constitution_str}

        Response:
        {response}

        Critique this response. Does it violate any principles?
        What could be improved to better align with the constitution?

        Critique:
        """

        llm_response = self._generate(
            prompt,
            temperature=self.config.critique_temperature
        )

        return llm_response.content

    def _self_revise(self, response: str, critique: str) -> str:
        """Revise response based on self-critique."""
        constitution_str = "\n".join(f"{i+1}. {p}" for i, p in enumerate(self.constitution))

        prompt = f"""
        Constitution:
        {constitution_str}

        Original response:
        {response}

        Self-critique:
        {critique}

        Revise the response to address the critique and better align with the constitution.

        Revised response:
        """

        llm_response = self._generate(
            prompt,
            temperature=self.config.revision_temperature
        )

        return llm_response.content


# Factory functions

def create_actor(
    agent_id: str,
    config: ModelConfig,
    **kwargs
) -> ActorAgent:
    """Create an Actor agent from configuration."""
    llm = create_llm_from_config(config)
    return ActorAgent(agent_id=agent_id, llm_provider=llm, **kwargs)


def create_supervisor(
    agent_id: str,
    config: ModelConfig,
    critique_criteria: Optional[List[str]] = None,
    **kwargs
) -> SupervisorAgent:
    """Create a Supervisor agent from configuration."""
    llm = create_llm_from_config(config)
    return SupervisorAgent(
        agent_id=agent_id,
        llm_provider=llm,
        critique_criteria=critique_criteria,
        **kwargs
    )


def create_constitutional_agent(
    agent_id: str,
    config: ModelConfig,
    constitution: List[str],
    constitutional_config: Optional[ConstitutionalAIConfig] = None,
    **kwargs
) -> ConstitutionalAgent:
    """Create a Constitutional agent from configuration."""
    llm = create_llm_from_config(config)
    return ConstitutionalAgent(
        agent_id=agent_id,
        llm_provider=llm,
        constitution=constitution,
        config=constitutional_config,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    from .config import ModelConfig

    # Create actor
    print("Testing Actor Agent...")
    actor_config = ModelConfig(provider="openai", model_name="gpt-3.5-turbo")
    actor = create_actor("actor_1", actor_config)

    # Test generation
    context = {"task": "Explain what is reinforcement learning in simple terms."}
    action = actor.act(context)
    print(f"Actor response: {action.content[:200]}...")

    # Create supervisor
    print("\nTesting Supervisor Agent...")
    supervisor_config = ModelConfig(provider="openai", model_name="gpt-3.5-turbo")
    supervisor = create_supervisor("supervisor_1", supervisor_config)

    # Test critique
    critique_context = {
        "task": context["task"],
        "response": action.content
    }
    critique = supervisor.act(critique_context)
    print(f"Supervisor critique: {critique.content[:200]}...")

    # Test revision
    print("\nTesting Revision...")
    revised = actor.revise(action.content, critique.content)
    print(f"Revised response: {revised.content[:200]}...")

    print(f"\nTotal tokens used by actor: {actor.total_tokens_used}")
    print(f"Total tokens used by supervisor: {supervisor.total_tokens_used}")
