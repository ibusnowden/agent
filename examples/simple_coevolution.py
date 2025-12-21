"""
Simple Co-Evolution Example

Demonstrates the core Actor → Supervisor → Judge loop.

This is "The Scrutiny Room" in action:
1. Actor generates a response
2. Supervisor critiques it
3. Judge evaluates
4. If not satisfactory, loop repeats with feedback
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    ModelConfig,
    create_actor,
    create_supervisor,
    create_llm_judge,
    CoEvolutionStateMachine,
)


def main():
    """Run a simple co-evolution example."""

    print("="*80)
    print("COEVOLVE: Simple Co-Evolution Example")
    print("="*80)

    # Configuration
    config = ModelConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=512
    )

    print("\n1. Creating agents...")
    # Create agents
    actor = create_actor("actor", config)
    supervisor = create_supervisor("supervisor", config)
    judge = create_llm_judge("judge", config)

    print("   ✓ Actor created")
    print("   ✓ Supervisor created")
    print("   ✓ Judge created")

    # Create state machine
    print("\n2. Creating co-evolution state machine...")
    state_machine = CoEvolutionStateMachine(
        actor=actor,
        supervisor=supervisor,
        judge=judge,
        max_iterations=3
    )
    print("   ✓ State machine ready")

    # Run co-evolution
    task = "Explain reinforcement learning to a 10-year-old."

    print(f"\n3. Running co-evolution on task:")
    print(f"   '{task}'")
    print("\n" + "-"*80)

    result = state_machine.run(task)

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\nIterations: {result.iterations}")
    print(f"Verdict: {result.judgment.verdict.value}")
    print(f"Confidence: {result.judgment.confidence:.2%}")
    print(f"Total tokens: {result.total_tokens}")

    print("\nFinal Response:")
    print("-"*80)
    print(result.final_response)
    print("-"*80)

    print("\nJudge's Reasoning:")
    print("-"*80)
    print(result.judgment.reasoning[:500] + "..." if len(result.judgment.reasoning) > 500 else result.judgment.reasoning)
    print("-"*80)

    print("\nIteration History:")
    for entry in result.history:
        print(f"  - {entry}")

    print("\n" + "="*80)
    print("Example complete!")
    print("="*80)


if __name__ == "__main__":
    main()
