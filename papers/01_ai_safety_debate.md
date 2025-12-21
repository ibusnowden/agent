# AI Safety via Debate

**Authors:** Geoffrey Irving, Paul Christiano, Dario Amodei
**Institution:** OpenAI
**Year:** 2018
**Paper:** [arxiv.org/abs/1805.00899](https://arxiv.org/abs/1805.00899)

---

## Core Idea

**Problem:** How can humans oversee AI systems that are smarter than us?

**Solution:** Use debate between two AI agents. A non-expert human judge can more easily identify truth when they see both sides of an argument, even if they couldn't generate the correct answer themselves.

**Key Insight:** Truth is an optimal strategy in debate. Lying is vulnerable to being exposed by an honest opponent.

---

## The Debate Game

### Setup

1. **Two AI Agents** (Alice and Bob) both try to convince a human judge
2. **The Question:** A question with a verifiable answer (e.g., "Is this image of a cat?")
3. **The Game:** Agents take turns making claims and counter-claims
4. **The Judge:** A human (who might not be an expert) picks the winner

### Rules

```
Round 1: Alice makes a claim
Round 2: Bob counters or concedes
Round 3: Alice responds to Bob
...
Round N: Judge picks the most convincing argument
```

### Why Truth Wins

- If Alice lies, Bob can point out the lie
- If both lie, the more subtle lie wins (but truth is most robust)
- Over many debates, truth is the only unbeatable strategy

---

## Game-Theoretic Analysis

### Nash Equilibrium

**Theorem:** In a perfect debate game with a perfect judge, the Nash equilibrium is for both agents to tell the truth.

**Proof Sketch:**
1. Assume Alice lies
2. Bob can expose the lie by pointing to contradictory evidence
3. Judge will favor Bob
4. Therefore, lying is not an equilibrium strategy

### Imperfect Judges

**Reality:** Human judges are not perfect. They have:
- Limited attention
- Cognitive biases
- Bounded rationality

**Question:** Does debate still work?

**Answer:** Yes, if:
1. Detecting a flaw is easier than finding the right answer
2. The debate tree is deep enough to expose lies
3. Agents have incentive to exploit opponent's lies

---

## Implementation Details

### Debate Tree Structure

```
Question: "What is the capital of France?"

Alice: "The capital is Paris."
  ├─ Bob: "That's correct." [Concede]
  └─ Bob: "No, it's Lyon."
      └─ Alice: "Here's a map showing Paris as capital."
          └─ Bob: "That map is from 1400 AD." [Caught lying]
```

### Judging Mechanism

**Simple Judge:**
- Reads the full debate
- Picks the most convincing side

**ML Judge:**
- Trained on (debate, outcome) pairs
- Learns what "good argument" looks like
- Can be the same model being debated (recursive oversight)

---

## Relation to COEVOLVE

### How We Use It

1. **Socratic Debater Game:** Direct implementation
   - Model A proposes answer
   - Model B critiques
   - Judge evaluates

2. **Training Signal:** Debate outcomes train models
   - Winning arguments → positive examples
   - Losing arguments → negative examples

3. **Scalable Oversight:** Human judges only final output
   - Models do the heavy lifting
   - Humans validate debate conclusions

### Novel Extensions

**Multi-Round Evolution:**
- Not just one debate, but generations of debates
- Losing models learn from winning models
- Memetic evolution of argument strategies

**Multi-Agent Debate:**
- More than 2 agents
- Coalition formation
- More complex game theory

**Grounded Judges:**
- Not just humans, but unit tests, compilers, math proofs
- Objective truth conditions reduce judge uncertainty

---

## Key Equations

### Debate Utility

For agent A making claim c:
```
U(c) = P(Judge picks A | c, opponent's response)
```

### Nash Equilibrium Condition

```
Truth is equilibrium if:
∀ lie l: U(truth) ≥ U(l)
```

This holds when:
```
P(opponent exposes l) × penalty > gain from lying
```

---

## Experimental Results (from paper)

### Image Classification Debate

**Setup:** Two models debate what's in an image. Judge sees only a 6×6 patch.

**Results:**
- Debate: 88% accuracy
- Direct question (no debate): 59% accuracy
- **Improvement: 29%** from debate

### Arithmetic Debate

**Setup:** Debate about arithmetic facts. Judge has limited compute.

**Results:**
- Debate converges to correct answer 94% of the time
- Direct question: 67% of the time

**Conclusion:** Debate amplifies weak judges.

---

## Strengths

1. **Theoretically Grounded:** Game theory provides formal guarantees
2. **Scalable:** Doesn't require expert human feedback on every question
3. **General:** Applies to any domain with verifiable answers
4. **Interpretable:** Humans see the reasoning process

---

## Limitations

1. **Assumes Good Judge:** If judge is too weak, debate can mislead
2. **Computational Cost:** Deep debate trees are expensive
3. **Adversarial Robustness:** Clever agents might still fool judges
4. **Subjective Questions:** Doesn't work well on opinion-based tasks

---

## Implementation Considerations

### For COEVOLVE

**What to implement:**
```python
class DebateGame:
    def __init__(self, question, model_a, model_b, judge):
        self.question = question
        self.model_a = model_a  # Proposer
        self.model_b = model_b  # Critic
        self.judge = judge      # Evaluator

    def run_debate(self, max_turns=4):
        debate_history = []

        # Turn 1: A proposes
        claim = self.model_a.generate(self.question)
        debate_history.append(("A", claim))

        for turn in range(max_turns):
            # B responds
            counter = self.model_b.generate(claim, debate_history)
            debate_history.append(("B", counter))

            # A responds
            rebuttal = self.model_a.generate(counter, debate_history)
            debate_history.append(("A", rebuttal))

        # Judge decides
        winner = self.judge.evaluate(debate_history)
        return winner, debate_history
```

**Judge implementation:**
```python
class Judge:
    def evaluate(self, debate_history):
        # Option 1: LLM Judge
        prompt = f"Who won this debate? {debate_history}"
        verdict = llm.generate(prompt)

        # Option 2: Grounded Judge (if available)
        # E.g., run the code and see if it works
        # E.g., check math proof

        return verdict
```

---

## Integration with Other Papers

**With Constitutional AI:**
- Debate generates preference pairs
- Winning argument = preferred
- Losing argument = rejected

**With STaR:**
- Successful debate arguments → training data
- Filter for "winning" reasoning chains
- Bootstrap better debaters

**With Voyager:**
- Store winning argument strategies as "skills"
- Retrieve relevant past debates
- Build library of robust arguments

---

## Research Questions for COEVOLVE

1. **Does debate improve over pure self-critique?**
   - Compare Constitutional AI vs. Debate
   - Measure: robustness to adversarial attacks

2. **What's the optimal debate depth?**
   - Trade-off: deeper debates vs. computational cost
   - Measure: judge accuracy vs. # turns

3. **Can models learn to be better judges through meta-learning?**
   - Train judge on debate outcomes
   - Measure: judge accuracy improvement over time

4. **Does multi-domain debate create transferable skills?**
   - Train on math debates, test on code debates
   - Measure: zero-shot performance

---

## Critical Insights for Implementation

### 1. Judge is the Bottleneck

The quality of oversight is bounded by judge quality. Invest in:
- Grounded judges (tests, verifiers)
- Multi-judge consensus
- Judge calibration

### 2. Debate Trees Grow Fast

A 4-turn debate with 3 responses per turn = 3^4 = 81 paths.

**Solution:** Pruning strategies
- Beam search
- Best-first search
- Early stopping on clear wins

### 3. Training from Debates

Don't just run debates, use them:
- Collect (question, winning_arg, losing_arg) triplets
- Train models to produce winning arguments
- This is the co-evolution step

---

## Pseudocode for Full Implementation

```python
def train_via_debate(model_a, model_b, dataset, num_epochs):
    """
    Co-evolutionary training via debate.
    """
    for epoch in range(num_epochs):
        preference_pairs = []

        for question in dataset:
            # Run debate
            winner, debate = run_debate(question, model_a, model_b)

            # Extract preference pairs
            if winner == "A":
                preferred = debate.get_a_arguments()
                rejected = debate.get_b_arguments()
            else:
                preferred = debate.get_b_arguments()
                rejected = debate.get_a_arguments()

            preference_pairs.append((question, preferred, rejected))

        # Train both models on preference pairs
        model_a = train_dpo(model_a, preference_pairs)
        model_b = train_dpo(model_b, preference_pairs)

    return model_a, model_b
```

---

## Conclusion

AI Safety via Debate provides the **game-theoretic foundation** for our co-evolutionary framework. It proves that:

1. Adversarial interaction can surface truth
2. Weak judges can oversee strong models
3. Debate is a training signal

**For COEVOLVE:** This is the core mechanism. All other papers build on this foundation.

---

## Next Steps

- [ ] Implement basic 2-agent debate
- [ ] Test with simple questions (math, logic)
- [ ] Measure judge accuracy vs. debate depth
- [ ] Integrate with DPO training (next paper)

---

## References

- Original Paper: https://arxiv.org/abs/1805.00899
- OpenAI Blog: https://openai.com/research/debate
- Follow-up: Khan et al. (2024) "Debate Helps Supervise Unreliable Experts"
