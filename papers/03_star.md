# STaR: Self-Taught Reasoner

**Authors:** Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman
**Institution:** Stanford University, Google Research
**Year:** 2022
**Paper:** [arxiv.org/abs/2203.14465](https://arxiv.org/abs/2203.14465)

---

## Core Idea

**Problem:** Chain-of-thought reasoning improves LLM performance, but collecting reasoning chains is expensive.

**Solution:** Bootstrap reasoning chains from the model's own successful attempts.

**Key Insight:** If the model gets the right answer (even by luck), the reasoning it used is probably good. Train on those successful reasoning chains.

---

## The STaR Algorithm

### High-Level Process

```
1. Generate reasoning chains for questions
2. Check if final answer is correct
3. If correct: Add reasoning chain to training set
4. If wrong: Generate rationale for correct answer, add that
5. Fine-tune model on collected reasoning chains
6. Repeat until convergence
```

### Why It Works

**Virtuous Cycle:**
- Better model → Better reasoning chains
- Better reasoning chains → Better training data
- Better training data → Better model

**Filtering is Key:**
- Only train on successful reasoning
- Removes hallucinations and errors
- Quality > Quantity

---

## Detailed Algorithm

### Phase 1: Rationalization

```python
def generate_rationales(model, question, answer):
    """
    Generate multiple reasoning chains.
    """
    rationales = []

    for _ in range(num_samples):
        prompt = f"Question: {question}\nLet's think step by step:"
        reasoning = model.generate(prompt)

        # Extract final answer from reasoning
        predicted_answer = extract_answer(reasoning)

        if predicted_answer == answer:
            # Success! Save this reasoning
            rationales.append((question, reasoning, answer))

    return rationales
```

### Phase 2: Rationalization with Hints (Optional)

If model can't solve problem, give it the answer and ask for reasoning:

```python
def generate_rationales_with_hints(model, question, answer):
    """
    Generate reasoning given the correct answer.
    """
    prompt = f"""
    Question: {question}
    The answer is: {answer}

    Let's work backwards. Why is this the answer?
    Reasoning:
    """
    reasoning = model.generate(prompt)
    return (question, reasoning, answer)
```

### Phase 3: Training

```python
def train_star(model, dataset, num_iterations=5):
    """
    Full STaR training loop.
    """
    for iteration in range(num_iterations):
        training_data = []

        for question, answer in dataset:
            # Generate rationales
            rationales = generate_rationales(model, question, answer)

            if len(rationales) > 0:
                # Use successful reasoning
                training_data.extend(rationales)
            else:
                # Use hints if needed
                hint_rationale = generate_rationales_with_hints(
                    model, question, answer
                )
                training_data.append(hint_rationale)

        # Fine-tune on collected rationales
        model = finetune(model, training_data)

        # Evaluate improvement
        accuracy = evaluate(model, test_set)
        print(f"Iteration {iteration}: Accuracy = {accuracy}")

    return model
```

---

## Key Components

### 1. Reasoning Chain Format

**Example:**

```
Question: What is 15% of 80?

Reasoning:
Let's solve this step by step:
1. Convert 15% to decimal: 15/100 = 0.15
2. Multiply by 80: 0.15 × 80
3. Calculate: 0.15 × 80 = 12

Answer: 12
```

### 2. Answer Extraction

```python
def extract_answer(reasoning_chain):
    """
    Extract final answer from reasoning chain.
    """
    # Look for "Answer:" or similar markers
    if "Answer:" in reasoning_chain:
        return reasoning_chain.split("Answer:")[-1].strip()

    # Or take last line
    return reasoning_chain.strip().split("\n")[-1]
```

### 3. Filtering Criterion

**Simple:** Answer matches ground truth

**Advanced:**
- Answer matches AND reasoning is valid
- Verify intermediate steps
- Check for logical consistency

---

## Experimental Results (from paper)

### CommonsenseQA

**Setup:** Multiple-choice commonsense reasoning questions.

**Results:**

| Method | Accuracy |
|--------|----------|
| Base Model (no reasoning) | 72.5% |
| Few-shot Chain-of-Thought | 78.2% |
| STaR (1 iteration) | 81.3% |
| STaR (5 iterations) | 82.9% |

**Improvement: +10.4% over base!**

### GSM8K (Math Word Problems)

**Results:**

| Method | Accuracy |
|--------|----------|
| Base Model | 12.7% |
| Few-shot CoT | 40.7% |
| STaR (with hints) | 58.4% |

**Improvement: +17.7% over few-shot CoT!**

---

## Relation to COEVOLVE

### How We Use It

1. **Debate Reasoning Chains**
   - Store winning debate arguments
   - Train models on successful debate strategies
   - Bootstrap better debaters

2. **Code Solution Library**
   - Store code that passes tests
   - Train on correct implementations
   - Build up skill library

3. **Hypothesis Generation**
   - Store hypotheses that get validated
   - Train on successful scientific reasoning
   - Accumulate domain knowledge

### Novel Extensions

**Multi-Agent STaR:**
- Model A generates reasoning
- Model B validates reasoning
- Only train on reasoning that B approves

**Adversarial STaR:**
- Include "near-miss" reasoning chains
- Explicitly train model to avoid subtle errors
- Use debate losers as negative examples

**Hierarchical STaR:**
- Low-level: Individual reasoning steps
- High-level: Overall problem-solving strategies
- Train on both levels

---

## Integration with Other Papers

**With Debate:**
- Winning debates → high-quality reasoning chains
- Losing debates → filtered out
- STaR trains on debate winners

**With Constitutional AI:**
- Reasoning must satisfy constitution
- Double filter: Correct answer AND constitutional
- Higher quality training data

**With Voyager:**
- Reasoning chains = skills
- Store in skill library
- Retrieve for similar problems

---

## Critical Insights

### 1. Bootstrapping Requires Diversity

**Problem:** Model generates same reasoning repeatedly.

**Solution:** Temperature sampling
```python
# Generate diverse reasoning chains
rationales = model.generate(prompt, temperature=0.7, n=10)
```

### 2. Hints Prevent Stagnation

**Problem:** Model can't solve hard problems initially.

**Solution:** Provide answer, ask for reasoning
```python
if model.solve(question) == wrong:
    reasoning = model.explain(question, correct_answer)
```

### 3. Iterative Improvement

**Observation:** Each iteration improves performance by ~2-5%.

**Implication:** Need multiple iterations, but diminishing returns after ~5.

**Strategy:**
- Start with easy problems
- Gradually increase difficulty
- Curriculum learning

---

## Implementation for COEVOLVE

### Core STaR Engine

```python
class STaRTrainer:
    def __init__(self, model, judge):
        self.model = model
        self.judge = judge  # Evaluates correctness
        self.training_buffer = []

    def collect_rationales(self, question, answer, num_samples=10):
        """
        Generate and filter reasoning chains.
        """
        successful_rationales = []

        for _ in range(num_samples):
            # Generate reasoning
            prompt = f"Question: {question}\nReasoning:"
            reasoning = self.model.generate(prompt, temperature=0.7)

            # Check if correct
            predicted = extract_answer(reasoning)
            if self.judge.evaluate(predicted, answer):
                successful_rationales.append(reasoning)

        return successful_rationales

    def collect_with_hints(self, question, answer):
        """
        Generate reasoning given the answer.
        """
        prompt = f"Question: {question}\nAnswer: {answer}\nExplain why:"
        reasoning = self.model.generate(prompt)
        return reasoning

    def train_iteration(self, dataset):
        """
        One iteration of STaR training.
        """
        training_data = []

        for question, answer in dataset:
            # Try to solve without hints
            rationales = self.collect_rationales(question, answer)

            if rationales:
                # Use successful attempts
                for r in rationales:
                    training_data.append((question, r, answer))
            else:
                # Use hints
                r = self.collect_with_hints(question, answer)
                training_data.append((question, r, answer))

        # Fine-tune
        self.model = finetune(self.model, training_data)

        # Evaluate
        accuracy = self.evaluate(dataset)
        return accuracy
```

### Multi-Agent STaR (Novel)

```python
class MultiAgentSTaR:
    def __init__(self, generator, validator, judge):
        self.generator = generator  # Model A: Generates reasoning
        self.validator = validator  # Model B: Validates reasoning
        self.judge = judge          # Ground truth checker

    def collect_rationales(self, question, answer):
        """
        Generate reasoning, validate it, then check correctness.
        """
        # Generator creates reasoning
        reasoning = self.generator.generate(question)

        # Validator checks reasoning quality
        validation_prompt = f"""
        Question: {question}
        Reasoning: {reasoning}

        Is this reasoning logically sound? (Yes/No)
        Critique:
        """
        validation = self.validator.generate(validation_prompt)

        # Extract verdict
        is_valid = "yes" in validation.lower()

        # Judge checks correctness
        predicted = extract_answer(reasoning)
        is_correct = self.judge.evaluate(predicted, answer)

        # Only store if both valid AND correct
        if is_valid and is_correct:
            return reasoning, validation
        else:
            return None

    def train_iteration(self, dataset):
        """
        Co-evolutionary STaR training.
        """
        generator_data = []
        validator_data = []

        for question, answer in dataset:
            result = self.collect_rationales(question, answer)

            if result:
                reasoning, validation = result

                # Train generator on good reasoning
                generator_data.append((question, reasoning, answer))

                # Train validator on good validation
                # (Validator learns what "good reasoning" looks like)
                validator_data.append((reasoning, "Valid", validation))

        # Update both models
        self.generator = finetune(self.generator, generator_data)
        self.validator = finetune(self.validator, validator_data)
```

---

## Research Questions for COEVOLVE

1. **Does multi-agent STaR beat single-agent?**
   - Compare: Self-validation vs. Peer validation
   - Measure: Quality of reasoning chains

2. **Optimal sampling strategy?**
   - Temperature vs. Top-p vs. Beam search
   - Measure: Diversity vs. quality trade-off

3. **When to use hints?**
   - Always vs. Only when stuck vs. Never
   - Measure: Learning efficiency

4. **Does STaR reasoning transfer?**
   - Train on math, test on code
   - Measure: Zero-shot reasoning quality

---

## Failure Modes and Mitigations

### 1. Reward Hacking

**Problem:** Model learns to produce correct answer without valid reasoning.

**Example:**
```
Question: What is 7 × 8?
"Reasoning": I just know it's 56.
Answer: 56 ✓
```

**Solution:** Validate intermediate steps
```python
def check_reasoning_steps(reasoning, answer):
    # Parse steps
    steps = extract_steps(reasoning)

    # Verify each step
    for step in steps:
        if not is_valid_step(step):
            return False

    # Check final answer
    return compute_from_steps(steps) == answer
```

### 2. Circular Reasoning

**Problem:** Model uses answer to generate "reasoning".

**Solution:** Blind validation
```python
# Validator sees reasoning but NOT the answer
validation = validator.validate(reasoning)  # No answer provided
```

### 3. Overfitting to Easy Examples

**Problem:** Model only learns to solve easy problems.

**Solution:** Curriculum learning
```python
def curriculum_star(model, dataset):
    # Sort by difficulty
    sorted_data = sort_by_difficulty(dataset)

    for difficulty_level in range(1, max_difficulty):
        # Train on current difficulty
        subset = sorted_data[difficulty_level]
        model = star_iteration(model, subset)

        # Test on next difficulty
        accuracy = evaluate(model, sorted_data[difficulty_level + 1])
```

---

## Pseudocode for Full Implementation

```python
def train_star_coevolve(generator, validator, dataset, num_iterations=5):
    """
    Full STaR training with co-evolution.
    """
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}")

        # Collect rationales
        rationale_data = []
        validation_data = []

        for question, answer in dataset:
            # Generate multiple attempts
            for attempt in range(10):
                reasoning = generator.generate(question, temperature=0.7)

                # Validate reasoning
                is_valid = validator.validate(reasoning)

                # Check correctness
                predicted = extract_answer(reasoning)
                is_correct = (predicted == answer)

                if is_valid and is_correct:
                    # Store successful reasoning
                    rationale_data.append((question, reasoning, answer))

                    # Store good validation
                    validation_data.append((reasoning, "Valid"))
                elif is_correct and not is_valid:
                    # Correct but invalid reasoning - teach validator
                    validation_data.append((reasoning, "Invalid"))

        # Train generator on good reasoning
        if rationale_data:
            generator = finetune(generator, rationale_data)

        # Train validator to recognize good reasoning
        if validation_data:
            validator = finetune(validator, validation_data)

        # Evaluate
        accuracy = evaluate(generator, test_set)
        print(f"Accuracy: {accuracy:.2%}")

    return generator, validator
```

---

## Metrics for Evaluation

### Reasoning Quality

```python
def evaluate_reasoning_quality(model, dataset):
    """
    Measure quality of reasoning chains.
    """
    metrics = {
        'correctness': [],
        'step_validity': [],
        'coherence': [],
        'completeness': []
    }

    for question, answer in dataset:
        reasoning = model.generate(question)

        # Correctness
        predicted = extract_answer(reasoning)
        metrics['correctness'].append(predicted == answer)

        # Step validity
        steps = extract_steps(reasoning)
        valid_steps = sum(is_valid_step(s) for s in steps) / len(steps)
        metrics['step_validity'].append(valid_steps)

        # Coherence (using validator)
        coherence = validator.score(reasoning)
        metrics['coherence'].append(coherence)

        # Completeness (are all necessary steps present?)
        completeness = check_completeness(reasoning, question)
        metrics['completeness'].append(completeness)

    return {k: np.mean(v) for k, v in metrics.items()}
```

---

## Conclusion

STaR provides the **bootstrapping mechanism** for our co-evolutionary framework. It proves that:

1. Models can learn from their own successful attempts
2. Filtering is more important than volume
3. Iterative improvement compounds over time
4. Hints can jumpstart learning

**For COEVOLVE:** This is the data generation engine. Combined with debate (selection) and constitutional AI (filtering), it creates a self-sustaining improvement loop.

---

## Next Steps

- [ ] Implement basic STaR loop
- [ ] Test with math and code problems
- [ ] Implement multi-agent variant
- [ ] Measure reasoning quality over iterations
- [ ] Integrate with debate and constitutional AI

---

## References

- Original Paper: https://arxiv.org/abs/2203.14465
- Code: https://github.com/ezelikman/STaR
- Follow-up: "Large Language Models can Self-Improve" (Huang et al., 2022)
