"""
Simple Debate Example

Demonstrates two actors debating while a judge evaluates.

This implements the "AI Safety via Debate" concept.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    ModelConfig,
    create_actor,
    create_llm_judge,
    DebateStateMachine,
)


def main():
    """Run a simple debate example."""

    print("="*80)
    print("COEVOLVE: Simple Debate Example")
    print("="*80)

    # Configuration
    config = ModelConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.8,  # Higher temp for more diverse arguments
        max_tokens=300
    )

    print("\n1. Creating debaters and judge...")
    # Create two actors with different system prompts
    actor_a = create_actor(
        "debater_for",
        config,
        system_prompt="You are debating FOR the topic. Present strong arguments in support."
    )

    actor_b = create_actor(
        "debater_against",
        config,
        system_prompt="You are debating AGAINST the topic. Present strong counter-arguments."
    )

    judge = create_llm_judge("debate_judge", config)

    print("   ✓ Debater FOR created")
    print("   ✓ Debater AGAINST created")
    print("   ✓ Judge created")

    # Create debate state machine
    print("\n2. Creating debate state machine...")
    debate = DebateStateMachine(
        actor_a=actor_a,
        actor_b=actor_b,
        judge=judge,
        max_turns=3
    )
    print("   ✓ Debate machine ready")

    # Run debate
    topic = "Artificial general intelligence will be beneficial for humanity."

    print(f"\n3. Starting debate on topic:")
    print(f"   '{topic}'")
    print("\n" + "-"*80)

    result = debate.run(topic)

    # Display results
    print("\n" + "="*80)
    print("DEBATE RESULTS")
    print("="*80)

    print(f"\nTurns: {result.iterations}")
    print(f"Winner: {result.judgment.verdict.value}")
    print(f"Confidence: {result.judgment.confidence:.2%}")
    print(f"Total tokens: {result.total_tokens}")

    print("\nDebate Transcript:")
    print("-"*80)
    for entry in result.history:
        agent_label = "FOR" if entry["agent"] == "actor_a" else "AGAINST"
        print(f"\n[Turn {entry['turn']}] {agent_label}:")
        print(entry["content"])
        print()
    print("-"*80)

    print("\nJudge's Final Verdict:")
    print("-"*80)
    print(result.judgment.reasoning)
    print("-"*80)

    print("\n" + "="*80)
    print(f"Debate complete! Winner: {result.judgment.verdict.value}")
    print("="*80)


if __name__ == "__main__":
    main()
