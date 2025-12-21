"""
COEVOLVE Core Infrastructure

This package provides the foundational components for multi-agent co-evolutionary learning.
"""

# Configuration
from .config import (
    CoEvolveConfig,
    ModelConfig,
    DebateConfig,
    CodeSecurityConfig,
    HypothesisConfig,
    MemoryConfig,
    STaRConfig,
    ConstitutionalAIConfig,
    DPOConfig,
    ExperimentConfig,
    get_quick_test_config,
    get_research_config,
    get_local_config,
)

# LLM Providers
from .llm_provider import (
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    LocalLLMProvider,
    LLMResponse,
    get_llm_provider,
    create_llm_from_config,
)

# Agents
from .agent import (
    BaseAgent,
    ActorAgent,
    SupervisorAgent,
    ConstitutionalAgent,
    AgentAction,
    AgentMemory,
    create_actor,
    create_supervisor,
    create_constitutional_agent,
)

# Judges
from .judge import (
    Judge,
    LLMJudge,
    GroundedJudge,
    CodeSecurityJudge,
    ConsensusJudge,
    MetaJudge,
    Verdict,
    JudgmentResult,
    create_llm_judge,
    create_code_security_judge,
    create_consensus_judge,
)

# State Machines
from .state_machine import (
    CoEvolutionStateMachine,
    DebateStateMachine,
    CoEvolutionState,
    CoEvolutionResult,
)

__all__ = [
    # Config
    'CoEvolveConfig',
    'ModelConfig',
    'DebateConfig',
    'CodeSecurityConfig',
    'HypothesisConfig',
    'MemoryConfig',
    'STaRConfig',
    'ConstitutionalAIConfig',
    'DPOConfig',
    'ExperimentConfig',
    'get_quick_test_config',
    'get_research_config',
    'get_local_config',

    # LLM Providers
    'BaseLLMProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'LocalLLMProvider',
    'LLMResponse',
    'get_llm_provider',
    'create_llm_from_config',

    # Agents
    'BaseAgent',
    'ActorAgent',
    'SupervisorAgent',
    'ConstitutionalAgent',
    'AgentAction',
    'AgentMemory',
    'create_actor',
    'create_supervisor',
    'create_constitutional_agent',

    # Judges
    'Judge',
    'LLMJudge',
    'GroundedJudge',
    'CodeSecurityJudge',
    'ConsensusJudge',
    'MetaJudge',
    'Verdict',
    'JudgmentResult',
    'create_llm_judge',
    'create_code_security_judge',
    'create_consensus_judge',

    # State Machines
    'CoEvolutionStateMachine',
    'DebateStateMachine',
    'CoEvolutionState',
    'CoEvolutionResult',
]
