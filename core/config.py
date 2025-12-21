"""
Configuration for COEVOLVE Framework

Centralized configuration for all experiments and components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for language models."""

    # Model selection
    provider: str = "openai"  # "openai", "anthropic", "huggingface", "local"
    model_name: str = "gpt-4"  # or "claude-3-opus", "llama-2-70b", etc.

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # API configuration
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 60


@dataclass
class DebateConfig:
    """Configuration for debate game."""

    max_turns: int = 4  # Maximum debate rounds
    turn_timeout: int = 120  # Seconds per turn

    # Exploration
    add_noise: bool = True  # Add exploration to root
    noise_alpha: float = 0.25

    # Judging
    judge_temperature: float = 0.3  # Lower temp for more deterministic judging
    require_reasoning: bool = True  # Judge must explain decision


@dataclass
class CodeSecurityConfig:
    """Configuration for Red Team/Blue Team code security game."""

    max_attempts: int = 3  # Max attempts for code generation
    test_timeout: int = 10  # Seconds to run tests

    # Security checks
    check_sql_injection: bool = True
    check_command_injection: bool = True
    check_xss: bool = True
    check_path_traversal: bool = True

    # Complexity
    max_code_length: int = 500  # Lines
    require_tests: bool = True


@dataclass
class HypothesisConfig:
    """Configuration for hypothesis builder game."""

    max_refinement_iterations: int = 5
    evidence_required: int = 3  # Minimum evidence pieces
    peer_review_count: int = 2  # Number of peer reviewers


@dataclass
class MemoryConfig:
    """Configuration for memory systems."""

    # Vector store
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db: str = "chromadb"  # "chromadb", "faiss", "pinecone"
    persist_directory: str = "data/memory"

    # Retrieval
    top_k: int = 5
    recency_weight: float = 1.0
    importance_weight: float = 1.0
    relevance_weight: float = 1.0

    # Reflection
    reflection_threshold: int = 100  # Importance score trigger
    reflection_window: int = 100  # Number of recent memories to reflect on

    # Pruning
    max_memories: int = 10000
    prune_strategy: str = "importance"  # "importance", "age", "hybrid"


@dataclass
class STaRConfig:
    """Configuration for Self-Taught Reasoner algorithm."""

    num_samples: int = 10  # Samples per question
    num_iterations: int = 5  # Training iterations
    use_hints: bool = True  # Use hints when stuck
    hint_threshold: float = 0.3  # If success rate < threshold, use hints

    # Filtering
    min_reasoning_length: int = 50  # Minimum characters in reasoning
    require_step_by_step: bool = True  # Must show steps

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-5


@dataclass
class ConstitutionalAIConfig:
    """Configuration for Constitutional AI."""

    # Constitutions
    constitution_file: Optional[str] = None
    default_principles: List[str] = field(default_factory=lambda: [
        "Be helpful, harmless, and honest",
        "Provide accurate information",
        "Avoid harmful or illegal content",
        "Show reasoning transparently"
    ])

    # Iteration
    num_critique_iterations: int = 3
    critique_temperature: float = 0.7
    revision_temperature: float = 0.8

    # Preference learning
    comparison_pairs: int = 1000  # Pairs to generate
    preference_model_type: str = "reward"  # "reward", "classifier"


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""

    # Training
    beta: float = 0.1  # KL divergence coefficient
    learning_rate: float = 5e-7
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Data
    preference_pairs: int = 1000
    max_length: int = 2048

    # Regularization
    weight_decay: float = 0.01
    warmup_steps: int = 100


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""

    # General
    name: str = "default_experiment"
    seed: int = 42
    num_runs: int = 1

    # Logging
    log_dir: str = "experiments/logs"
    save_dir: str = "experiments/results"
    log_interval: int = 10  # Log every N steps
    save_interval: int = 100  # Save checkpoints every N steps

    # Evaluation
    eval_interval: int = 50
    eval_episodes: int = 10

    # Parallelization
    num_workers: int = 1
    use_gpu: bool = False


@dataclass
class CoEvolveConfig:
    """Master configuration for COEVOLVE framework."""

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    code_security: CodeSecurityConfig = field(default_factory=CodeSecurityConfig)
    hypothesis: HypothesisConfig = field(default_factory=HypothesisConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    star: STaRConfig = field(default_factory=STaRConfig)
    constitutional: ConstitutionalAIConfig = field(default_factory=ConstitutionalAIConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Co-evolution specific
    num_generations: int = 10  # Number of co-evolution cycles
    evolution_strategy: str = "adversarial"  # "adversarial", "cooperative", "mixed"

    # Model assignment
    actor_model: ModelConfig = field(default_factory=ModelConfig)
    supervisor_model: ModelConfig = field(default_factory=ModelConfig)
    judge_model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        """Initialize after creation."""
        # Create directories
        Path(self.memory.persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.experiment.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.experiment.save_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CoEvolveConfig':
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # TODO: Implement recursive dataclass initialization
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict

        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# Preset configurations for common scenarios

def get_quick_test_config() -> CoEvolveConfig:
    """Get configuration for quick testing (small models, few iterations)."""
    config = CoEvolveConfig()
    config.model.model_name = "gpt-3.5-turbo"
    config.model.max_tokens = 512
    config.debate.max_turns = 2
    config.star.num_iterations = 2
    config.star.num_samples = 3
    config.num_generations = 3
    return config


def get_research_config() -> CoEvolveConfig:
    """Get configuration for full research experiments."""
    config = CoEvolveConfig()
    config.model.model_name = "gpt-4"
    config.model.max_tokens = 4096
    config.debate.max_turns = 6
    config.star.num_iterations = 10
    config.star.num_samples = 20
    config.num_generations = 20
    return config


def get_local_config() -> CoEvolveConfig:
    """Get configuration for local models (e.g., Llama)."""
    config = CoEvolveConfig()
    config.model.provider = "local"
    config.model.model_name = "llama-2-13b"
    config.model.api_base = "http://localhost:8000"
    return config


if __name__ == "__main__":
    # Example: Create and save configuration
    config = get_research_config()
    config.to_yaml("experiments/config_research.yaml")
    print("Configuration saved!")

    # Example: Load configuration
    loaded_config = CoEvolveConfig.from_yaml("experiments/config_research.yaml")
    print(f"Loaded config: {loaded_config.experiment.name}")
