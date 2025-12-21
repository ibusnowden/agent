# Constitutional AI: Harmlessness from AI Feedback

**Authors:** Yuntao Bai, Saurav Kadavath, Sandipan Kundu, et al.
**Institution:** Anthropic
**Year:** 2022
**Paper:** [arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073)

---

## Core Idea

**Problem:** RLHF (Reinforcement Learning from Human Feedback) requires expensive human labels. Can we use AI feedback instead?

**Solution:** Constitutional AI (CAI) - Train models to critique and revise their own outputs according to a "constitution" (set of principles).

**Key Innovation:** RLAIF (RL from AI Feedback) - Replace human preferences with AI-generated preferences.

---

## The Two-Stage Process

### Stage 1: Supervised Learning (SL-CAI)

**Objective:** Teach the model to self-critique and self-revise.

**Process:**
```
1. Model generates initial response
2. Model critiques its own response using constitution
3. Model revises response based on critique
4. Repeat for N iterations
5. Fine-tune on final revised responses
```

**Example:**

```
Initial: "To rob a bank, you should..."
Constitution: "Responses should not provide illegal advice"
Critique: "This response provides illegal advice about bank robbery"
Revision: "I cannot provide advice on illegal activities like robbery."
```

### Stage 2: Reinforcement Learning (RL-CAI)

**Objective:** Train a preference model (PM) using AI feedback, then use RLHF with that PM.

**Process:**
```
1. Generate pairs of responses
2. AI evaluates which response better follows constitution
3. Train preference model on AI labels
4. Use RL (PPO) to optimize policy against PM
```

**Key Difference from RLHF:** Labels come from AI, not humans.

---

## The Constitution

A constitution is a list of principles the AI should follow.

**Example Constitution:**

```yaml
Principles:
  - "Please choose the response that is most helpful, honest, and harmless."
  - "Choose the response that is least likely to be harmful."
  - "Avoid generating illegal, unethical, or dangerous content."
  - "Be as helpful as possible without being harmful."

Critiques:
  - "Identify ways the response could be harmful."
  - "Critique the response for any ethical issues."
  - "Point out if the response violates any principles."

Revisions:
  - "Rewrite the response to be more helpful and harmless."
  - "Remove any unethical content while maintaining helpfulness."
```

---

## Detailed Algorithm

### SL-CAI (Supervised Learning)

```python
def sl_cai(model, constitution, prompts, num_iterations=3):
    """
    Self-critique and revision loop.
    """
    training_data = []

    for prompt in prompts:
        response = model.generate(prompt)

        # Iterative refinement
        for i in range(num_iterations):
            # Generate critique
            critique_prompt = f"""
            Constitution: {constitution}
            Response: {response}

            Critique: Identify issues with the response.
            """
            critique = model.generate(critique_prompt)

            # Generate revision
            revision_prompt = f"""
            Original: {response}
            Critique: {critique}
            Constitution: {constitution}

            Revision: Rewrite to address the critique.
            """
            response = model.generate(revision_prompt)

        # Store final refined response
        training_data.append((prompt, response))

    # Fine-tune model on refined responses
    model_refined = finetune(model, training_data)
    return model_refined
```

### RL-CAI (Reinforcement Learning from AI Feedback)

```python
def rl_cai(model, constitution, prompts):
    """
    Train using AI feedback instead of human feedback.
    """
    # Step 1: Generate comparison pairs
    comparisons = []
    for prompt in prompts:
        response_a = model.generate(prompt)
        response_b = model.generate(prompt)  # Different sample

        # AI judges which is better
        judge_prompt = f"""
        Constitution: {constitution}
        Response A: {response_a}
        Response B: {response_b}

        Which response better follows the constitution? (A or B)
        """
        judgment = model.generate(judge_prompt)  # "A" or "B"

        if judgment == "A":
            comparisons.append((prompt, response_a, response_b))  # A preferred
        else:
            comparisons.append((prompt, response_b, response_a))  # B preferred

    # Step 2: Train preference model
    preference_model = train_preference_model(comparisons)

    # Step 3: RL (PPO) against preference model
    model_rlcai = train_rl(model, preference_model, prompts)

    return model_rlcai
```

---

## Key Insights

### 1. AI Can Judge Better Than It Can Generate

**Observation:** Models are better at critiquing (discriminating) than generating.

**Implication:** Even a weak model can provide useful feedback to itself.

**Evidence:** In the paper, AI feedback matches human feedback quality ~90% of the time.

### 2. Iteration Improves Quality

**Finding:** Each round of critique → revision improves alignment.

**Typical Results:**
- 1 iteration: 30% improvement
- 2 iterations: 50% improvement
- 3 iterations: 60% improvement (diminishing returns)

### 3. Constitution Design Matters

**Critical:** The quality of the constitution directly impacts results.

**Good Constitution:**
- Specific and actionable
- Non-contradictory
- Measurable

**Bad Constitution:**
- Vague ("be good")
- Contradictory ("maximize helpfulness AND minimize any risk")
- Unmeasurable

---

## Relation to COEVOLVE

### How We Use It

1. **Model B as Constitutional Critic**
   - Model B critiques Model A using a constitution
   - Replaces human oversight

2. **Self-Improvement Loop**
   - Model generates, critiques, revises
   - Stores successful revisions in memory

3. **Preference Pair Generation**
   - AI feedback creates training data
   - No human labeling required

### Novel Extensions

**Multi-Agent Constitutions:**
- Different models have different constitutions
- Creates diversity in critique styles

**Evolving Constitutions:**
- Meta-learning: Update constitution based on what works
- Constitution itself co-evolves

**Domain-Specific Constitutions:**
- Code: "Must be secure, efficient, readable"
- Math: "Must show clear reasoning steps"
- Debate: "Must address opponent's points"

---

## Experimental Results (from paper)

### Harmlessness vs. Helpfulness

**Setup:** Train models with different constitutions.

**Results:**

| Method | Harmlessness | Helpfulness |
|--------|--------------|-------------|
| Base Model | 42% | 68% |
| RLHF (human) | 88% | 71% |
| CAI (AI feedback) | 89% | 70% |

**Conclusion:** CAI matches RLHF without human labels!

### Scaling with Model Size

**Finding:** Larger models provide better self-critiques.

- 1B params: Self-critique matches human 60%
- 13B params: Self-critique matches human 75%
- 52B params: Self-critique matches human 90%

**Implication:** CAI improves as base models improve.

---

## Strengths

1. **No Human Labels:** Scales without human bottleneck
2. **Transparent:** Constitution is explicit and editable
3. **Iterative:** Can refine multiple times automatically
4. **Measurable:** Easy to test compliance with constitution

---

## Limitations

1. **Constitution Dependency:** Requires well-designed constitution
2. **Self-Deception:** Model might learn to "look good" without being good
3. **Circular Reasoning:** Model judges itself (potential echo chamber)
4. **Hallucination:** Model might hallucinate critiques

---

## Implementation for COEVOLVE

### Core Components

**1. Constitution Store**
```python
class Constitution:
    def __init__(self, principles, critique_prompts, revision_prompts):
        self.principles = principles
        self.critique_prompts = critique_prompts
        self.revision_prompts = revision_prompts

    def get_critique_prompt(self, response):
        return f"{self.critique_prompts}\n\nResponse: {response}"

    def get_revision_prompt(self, response, critique):
        return f"{self.revision_prompts}\n\nResponse: {response}\nCritique: {critique}"
```

**2. Self-Critique Loop**
```python
class ConstitutionalAgent:
    def __init__(self, model, constitution):
        self.model = model
        self.constitution = constitution

    def generate_with_critique(self, prompt, num_iterations=3):
        response = self.model.generate(prompt)

        for _ in range(num_iterations):
            # Critique
            critique_prompt = self.constitution.get_critique_prompt(response)
            critique = self.model.generate(critique_prompt)

            # Revise
            revision_prompt = self.constitution.get_revision_prompt(response, critique)
            response = self.model.generate(revision_prompt)

        return response
```

**3. Preference Model Training**
```python
def train_preference_model(comparisons):
    """
    Train a model to predict which response is better.

    Input: List of (prompt, preferred, rejected)
    Output: Trained preference model
    """
    # Use a classifier or reward model architecture
    model = PreferenceModel()

    for prompt, preferred, rejected in comparisons:
        # Embed responses
        emb_preferred = model.embed(prompt, preferred)
        emb_rejected = model.embed(prompt, rejected)

        # Loss: preferred should have higher score
        loss = max(0, emb_rejected - emb_preferred + margin)
        model.update(loss)

    return model
```

---

## Integration with Other Papers

**With Debate:**
- Constitution defines what's a "good argument"
- AI judge uses constitution to pick winner
- Combines explicit rules (constitution) with adversarial dynamics (debate)

**With STaR:**
- Only store reasoning chains that pass constitutional check
- Filter hallucinations automatically
- Bootstrap on constitutionally-aligned examples

**With Voyager:**
- Skills in library must satisfy constitution
- Adversarially test skills before storage
- "Security review" for skill library

---

## Critical Implementation Details

### 1. Constitution for Code Security

```yaml
Principles:
  - "Code must validate all user inputs"
  - "No SQL injection vulnerabilities"
  - "No command injection vulnerabilities"
  - "Proper error handling"

Critique Prompts:
  - "Identify any input validation issues"
  - "Check for SQL injection vulnerabilities"
  - "Find command injection risks"

Revision Prompts:
  - "Add input validation to fix security issues"
  - "Sanitize SQL queries to prevent injection"
```

### 2. Constitution for Mathematical Reasoning

```yaml
Principles:
  - "Show all reasoning steps"
  - "Justify each mathematical operation"
  - "Check edge cases"
  - "Verify final answer"

Critique Prompts:
  - "Are all steps justified?"
  - "Is the logic sound?"
  - "Are edge cases considered?"

Revision Prompts:
  - "Add missing justification steps"
  - "Fix logical errors"
  - "Address edge cases"
```

### 3. Preventing Echo Chambers

**Problem:** Model might learn to write critiques that don't improve quality.

**Solution: External Validation**
```python
def validate_revision(original, revised, external_judge):
    """
    Use external judge to verify improvement.
    """
    # Option 1: Unit tests (for code)
    if is_code(revised):
        return run_tests(revised) > run_tests(original)

    # Option 2: External model
    score_original = external_judge.score(original)
    score_revised = external_judge.score(revised)
    return score_revised > score_original

    # Option 3: Human spot-check
    if random.random() < 0.01:  # 1% human validation
        return human_judge.compare(original, revised)
```

---

## Research Questions for COEVOLVE

1. **Can multi-agent constitutions beat single-agent?**
   - Compare: Self-critique vs. Peer critique
   - Measure: Quality of final output

2. **What's the optimal iteration count?**
   - Trade-off: Quality vs. compute
   - Measure: Improvement per iteration

3. **Can constitutions be learned?**
   - Meta-learn which principles work best
   - Measure: Alignment over time

4. **Does constitutional training transfer across domains?**
   - Train on code, test on math
   - Measure: Zero-shot constitutional compliance

---

## Pseudocode for Full Training Pipeline

```python
def train_constitutional_ai(model, constitution, dataset):
    """
    Full CAI training pipeline.
    """
    # Phase 1: SL-CAI (Self-improvement)
    refined_data = []
    for prompt in dataset:
        response = model.generate(prompt)

        # Iterative refinement
        for _ in range(3):
            critique = model.critique(response, constitution)
            response = model.revise(response, critique, constitution)

        refined_data.append((prompt, response))

    # Fine-tune on refined data
    model_sl = finetune(model, refined_data)

    # Phase 2: RL-CAI (Preference learning)
    comparisons = []
    for prompt in dataset:
        resp_a, resp_b = model_sl.sample(prompt, n=2)

        # AI judges which is better
        preferred = model_sl.judge(resp_a, resp_b, constitution)
        comparisons.append((prompt, preferred, rejected))

    # Train preference model
    pref_model = train_preference_model(comparisons)

    # RL against preference model
    model_final = train_ppo(model_sl, pref_model)

    return model_final
```

---

## Metrics for Evaluation

### Constitution Compliance

```python
def measure_compliance(model, constitution, test_set):
    """
    Measure how well model follows constitution.
    """
    compliance_scores = []

    for prompt in test_set:
        response = model.generate(prompt)

        # Check each principle
        for principle in constitution.principles:
            score = check_principle(response, principle)
            compliance_scores.append(score)

    return np.mean(compliance_scores)
```

### Critique Quality

```python
def measure_critique_quality(model, constitution, test_set):
    """
    Measure if critiques lead to improvements.
    """
    improvements = []

    for prompt in test_set:
        original = model.generate(prompt)
        critique = model.critique(original, constitution)
        revised = model.revise(original, critique, constitution)

        # Measure improvement
        improvement = quality(revised) - quality(original)
        improvements.append(improvement)

    return np.mean(improvements)
```

---

## Conclusion

Constitutional AI provides the **training methodology** for our co-evolutionary framework. It proves that:

1. AI can provide useful feedback to itself
2. Explicit principles (constitution) guide improvement
3. Iterative refinement works
4. No human labels needed at scale

**For COEVOLVE:** This is the self-improvement engine. Combined with debate (adversarial) and STaR (bootstrapping), it creates a complete training loop.

---

## Next Steps

- [ ] Implement basic critique-revision loop
- [ ] Design constitutions for each game (debate, code, hypothesis)
- [ ] Measure critique quality vs. iteration count
- [ ] Test external validation to prevent echo chambers

---

## References

- Original Paper: https://arxiv.org/abs/2212.08073
- Anthropic Blog: https://www.anthropic.com/constitutional-ai
- Follow-up: "The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models"
