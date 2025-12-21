# Voyager: An Open-Ended Embodied Agent with Large Language Models

**Authors:** Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anand

konda
**Institution:** NVIDIA, Caltech, UT Austin, Stanford
**Year:** 2023
**Paper:** [arxiv.org/abs/2305.16291](https://arxiv.org/abs/2305.16291)

---

## Core Idea

**Problem:** Embodied agents (e.g., in Minecraft) struggle with open-ended exploration. They forget solutions and can't build on past successes.

**Solution:** Give agents a skill library - a growing collection of reusable code modules learned from experience.

**Key Innovation:** Agents write executable code (not natural language) and store successful programs as composable skills.

---

## The Three-Component Architecture

```
┌─────────────────────────────────────────────┐
│           VOYAGER ARCHITECTURE              │
│                                             │
│  ┌─────────────────┐                       │
│  │  Skill Library  │◄────┐                 │
│  │  (Code Storage) │     │                 │
│  └─────────────────┘     │                 │
│          │               │                 │
│          │ retrieve      │ store           │
│          ▼               │                 │
│  ┌─────────────────┐    │                 │
│  │  LLM Agent      │────┘                 │
│  │  (Code Writer)  │                      │
│  └─────────────────┘                      │
│          │                                 │
│          │ execute                         │
│          ▼                                 │
│  ┌─────────────────┐                      │
│  │  Environment    │                      │
│  │  (Minecraft)    │                      │
│  └─────────────────┘                      │
│          │                                 │
│          │ feedback                        │
│          ▼                                 │
│  ┌─────────────────┐                      │
│  │  Curriculum     │                      │
│  │  (Goal Setter)  │                      │
│  └─────────────────┘                      │
└─────────────────────────────────────────────┘
```

### 1. Automatic Curriculum

**What:** The agent proposes its own goals based on current state.

**Why:** Human-designed curricula don't adapt to agent's current capabilities.

**How:**
```python
def propose_next_task(agent_state, skill_library, previous_tasks):
    prompt = f"""
    Current state: {agent_state}
    Skills mastered: {skill_library.list_skills()}
    Recent tasks: {previous_tasks}

    Propose a new task that:
    1. Is achievable with current skills
    2. Expands exploration
    3. Is not too easy (already mastered)
    4. Is not too hard (impossible)

    Task:
    """
    return llm.generate(prompt)
```

**Example Progression:**
```
Task 1: Collect wood (basic)
Task 2: Craft wooden tools (builds on task 1)
Task 3: Mine stone with wooden pickaxe (builds on task 2)
Task 4: Build shelter (combines previous skills)
Task 5: Find iron ore (exploration + mining)
```

### 2. Skill Library

**What:** A vector database of executable code modules.

**Format:**
```python
@dataclass
class Skill:
    name: str
    code: str  # Executable JavaScript (for Minecraft API)
    description: str
    dependencies: List[str]  # Other skills this uses
    embedding: np.array  # For similarity search
    success_count: int
    usage_count: int
```

**Example Skill:**
```javascript
// Skill: mineBlock
async function mineBlock(bot, blockType, maxDistance = 32) {
    // Find nearest block of type
    const block = bot.findBlock({
        matching: mcData.blocksByName[blockType].id,
        maxDistance: maxDistance
    });

    if (!block) {
        return false;
    }

    // Equip appropriate tool
    await equipBestTool(bot, block);

    // Mine the block
    await bot.dig(block);

    return true;
}
```

**Retrieval:**
```python
def retrieve_skills(task, skill_library, top_k=5):
    """Retrieve relevant skills for a task."""
    task_embedding = embed(task)

    # Similarity search
    skills = skill_library.search(task_embedding, top_k)

    return skills
```

### 3. Iterative Prompting

**What:** The agent writes code, executes it, gets feedback, and revises.

**Process:**
```
1. Retrieve relevant skills from library
2. Generate code using LLM
3. Execute code in environment
4. Collect execution trace (success/failure, errors)
5. If failed: Self-debug using error messages
6. If succeeded: Store as new skill
```

**Self-Debugging:**
```python
def self_debug(code, error, max_retries=3):
    for attempt in range(max_retries):
        prompt = f"""
        Code:
        {code}

        Error:
        {error}

        Fix the code to resolve this error.
        Fixed code:
        """
        code = llm.generate(prompt)

        # Try executing again
        success, error = execute(code)

        if success:
            return code, True

    return code, False
```

---

## Detailed Algorithm

### Main Loop

```python
def voyager_loop(agent, environment):
    curriculum = AutomaticCurriculum()
    skill_library = SkillLibrary()

    while True:
        # 1. Propose next task
        task = curriculum.propose_task(
            agent.state,
            skill_library.list_skills(),
            curriculum.completed_tasks
        )

        # 2. Retrieve relevant skills
        relevant_skills = skill_library.retrieve(task, top_k=5)

        # 3. Generate code
        code = agent.write_code(task, relevant_skills)

        # 4. Execute code
        success, execution_info = environment.execute(code)

        # 5. Self-debug if needed
        if not success:
            code, success = agent.self_debug(code, execution_info.error)

        # 6. Update skill library
        if success:
            skill = Skill(
                name=task,
                code=code,
                description=task,
                embedding=embed(task)
            )
            skill_library.add(skill)
            curriculum.mark_completed(task)
        else:
            curriculum.mark_failed(task)
```

---

## Experimental Results (from paper)

### Exploration in Minecraft

**Setup:** 30-minute exploration sessions.

**Metrics:**
- Unique items obtained
- Distance traveled
- Complex items crafted

**Results:**

| Method | Unique Items | Distance (blocks) | Complex Items |
|--------|--------------|-------------------|---------------|
| Voyager | 142 | 3,890 | 38 |
| ReAct (no skills) | 29 | 1,234 | 4 |
| Reflexion | 67 | 2,100 | 12 |
| AutoGPT | 53 | 1,567 | 9 |

**Voyager achieves 2-5x better exploration!**

### Skill Composition

**Finding:** Voyager composes simple skills into complex behaviors.

**Example Chain:**
```
mineWood() → craftTable() → craftSticks() →
craftWoodenPickaxe() → mineStone() →
craftStonePickaxe() → mineIron()
```

This 7-step chain was learned automatically.

### Zero-Shot Transfer

**Setup:** Train in one Minecraft world, test in a different biome.

**Result:** 80% of skills transferred successfully to new environment.

---

## Relation to COEVOLVE

### How We Use It

1. **Code Library for Red Team/Blue Team**
   - Store secure coding patterns
   - Store exploit patterns (for red team)
   - Retrieve for similar security scenarios

2. **Reasoning Pattern Library**
   - Store successful debate arguments (as templates)
   - Store proof strategies (for math)
   - Retrieve for similar logical structures

3. **Hypothesis Templates**
   - Store validated hypothesis structures
   - Store experimental designs
   - Retrieve for similar scientific questions

### Novel Extensions

**Adversarial Skill Library:**
- Agent A stores attack skills
- Agent B stores defense skills
- Both evolve together

**Hierarchical Skills:**
- Low-level: Basic operations
- Mid-level: Common patterns
- High-level: Full solutions
- Composition at multiple levels

**Skill Versioning:**
- Track skill evolution over time
- Compare old vs. new implementations
- A/B test different versions

---

## Integration with Other Papers

**With Debate:**
- Debate arguments = skills
- Winning arguments stored in library
- Retrieve for similar debate topics

**With Constitutional AI:**
- Skills must satisfy constitution before storage
- "Security review" for each skill
- Adversarial testing before acceptance

**With STaR:**
- Reasoning chains = skills
- Store successful reasoning patterns
- Retrieve for similar problems

**With Generative Agents:**
- Skills = specialized memories
- Same retrieval mechanism (recency, importance, relevance)
- Reflection on skill effectiveness

---

## Implementation for COEVOLVE

### Core Skill Library

```python
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class Skill:
    """A reusable code/pattern module."""
    name: str
    code: str
    description: str
    dependencies: List[str]
    embedding: np.array
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime = None

class SkillLibrary:
    """Vector database of skills."""

    def __init__(self, embedding_model=None):
        if embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedding_model = embedding_model

        self.skills: Dict[str, Skill] = {}

    def add(self, skill: Skill):
        """Add a skill to the library."""
        # Embed description
        if skill.embedding is None:
            skill.embedding = self.embedding_model.encode(skill.description)

        # Set creation time
        if skill.created_at is None:
            skill.created_at = datetime.now()

        self.skills[skill.name] = skill

    def retrieve(self, query: str, top_k: int = 5) -> List[Skill]:
        """Retrieve most relevant skills."""
        query_embedding = self.embedding_model.encode(query)

        # Compute similarity scores
        scored_skills = []
        for skill in self.skills.values():
            similarity = np.dot(query_embedding, skill.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(skill.embedding)
            )

            # Boost frequently successful skills
            success_rate = skill.success_count / max(1, skill.success_count + skill.failure_count)
            score = similarity * (0.7 + 0.3 * success_rate)

            scored_skills.append((score, skill))

        # Sort and return top-k
        scored_skills.sort(reverse=True, key=lambda x: x[0])
        return [skill for score, skill in scored_skills[:top_k]]

    def record_success(self, skill_name: str):
        """Record successful use of a skill."""
        if skill_name in self.skills:
            self.skills[skill_name].success_count += 1

    def record_failure(self, skill_name: str):
        """Record failed use of a skill."""
        if skill_name in self.skills:
            self.skills[skill_name].failure_count += 1

    def list_skills(self) -> List[str]:
        """List all skill names."""
        return list(self.skills.keys())

    def get_skill(self, name: str) -> Skill:
        """Retrieve specific skill by name."""
        return self.skills.get(name)
```

### Automatic Curriculum

```python
class AutomaticCurriculum:
    """Proposes increasingly complex tasks."""

    def __init__(self, llm):
        self.llm = llm
        self.completed_tasks = []
        self.failed_tasks = []
        self.current_difficulty = 1

    def propose_task(self, agent_state, available_skills):
        """Propose next task based on current state."""
        prompt = f"""
        Agent State:
        {agent_state}

        Available Skills:
        {', '.join(available_skills)}

        Recently Completed:
        {', '.join(self.completed_tasks[-5:])}

        Recently Failed:
        {', '.join(self.failed_tasks[-3:])}

        Propose a new task that:
        1. Uses 1-3 existing skills
        2. Is slightly more challenging than recent tasks
        3. Avoids recently failed tasks
        4. Explores new capabilities

        Task (one sentence):
        """

        task = self.llm.generate(prompt).strip()
        return task

    def mark_completed(self, task):
        """Mark task as successfully completed."""
        self.completed_tasks.append(task)
        self.current_difficulty += 0.1

    def mark_failed(self, task):
        """Mark task as failed."""
        self.failed_tasks.append(task)
```

### Code Generation with Skills

```python
class CodeGeneratingAgent:
    """Agent that writes code using skill library."""

    def __init__(self, llm, skill_library):
        self.llm = llm
        self.skill_library = skill_library

    def write_code(self, task, max_attempts=3):
        """Generate code to accomplish task."""
        # Retrieve relevant skills
        relevant_skills = self.skill_library.retrieve(task, top_k=5)

        # Build context from skills
        skills_context = "\n\n".join([
            f"# Skill: {s.name}\n{s.code}" for s in relevant_skills
        ])

        # Generate code
        prompt = f"""
        Available Skills:
        {skills_context}

        Task: {task}

        Write code to accomplish this task using the available skills.
        Code:
        """

        code = self.llm.generate(prompt)

        return code

    def self_debug(self, code, error, task, max_retries=3):
        """Fix code based on error message."""
        for attempt in range(max_retries):
            prompt = f"""
            Task: {task}

            Code:
            {code}

            Error:
            {error}

            Fix the code. Return only the corrected code.
            Fixed Code:
            """

            code = self.llm.generate(prompt)

            # Would execute and check here
            # For now, just return fixed code

        return code
```

---

## Novel: Adversarial Skill Library

```python
class AdversarialSkillLibrary:
    """
    Two-player skill library for co-evolution.
    Red team (attack) vs Blue team (defense).
    """

    def __init__(self, embedding_model=None):
        self.red_library = SkillLibrary(embedding_model)   # Attack skills
        self.blue_library = SkillLibrary(embedding_model)  # Defense skills
        self.battles = []  # History of red vs blue

    def battle(self, task, red_agent, blue_agent):
        """
        Red team generates attack, blue team generates defense.
        """
        # Red team: Generate attack code
        attack_skills = self.red_library.retrieve(task, top_k=5)
        attack_code = red_agent.write_code(task, attack_skills)

        # Blue team: Generate defense code
        defense_skills = self.blue_library.retrieve(task, top_k=5)
        defense_code = blue_agent.write_code(task, defense_skills)

        # Judge: Does attack succeed despite defense?
        attack_succeeds = self.judge_battle(attack_code, defense_code)

        # Update libraries
        if attack_succeeds:
            # Attack won - store attack, blue team learns
            attack_skill = Skill(
                name=f"attack_{task}",
                code=attack_code,
                description=f"Attack: {task}"
            )
            self.red_library.add(attack_skill)
            self.red_library.record_success(attack_skill.name)
        else:
            # Defense won - store defense, red team learns
            defense_skill = Skill(
                name=f"defense_{task}",
                code=defense_code,
                description=f"Defense: {task}"
            )
            self.blue_library.add(defense_skill)
            self.blue_library.record_success(defense_skill.name)

        # Record battle
        self.battles.append({
            'task': task,
            'attack': attack_code,
            'defense': defense_code,
            'winner': 'red' if attack_succeeds else 'blue'
        })

        return attack_succeeds

    def judge_battle(self, attack_code, defense_code):
        """
        Determine if attack succeeds.
        Execute code and check for vulnerabilities.
        """
        # This would actually execute code and check security
        # For now, placeholder
        pass
```

---

## Research Questions for COEVOLVE

1. **Does skill library accelerate co-evolution?**
   - Compare: With skills vs. Without skills
   - Measure: Time to achieve performance threshold

2. **Optimal skill granularity?**
   - Low-level (single operations) vs. High-level (full solutions)
   - Measure: Reusability vs. specificity

3. **Skill versioning vs. replacement?**
   - Keep all versions vs. Replace with better version
   - Measure: Library size vs. performance

4. **Transfer across domains?**
   - Code skills → Math skills → Debate skills
   - Measure: Zero-shot performance on new domain

---

## Critical Implementation Details

### 1. Skill Deduplication

**Problem:** Agent might create duplicate skills.

**Solution:** Similarity check before adding
```python
def add_skill_with_dedup(self, new_skill, threshold=0.9):
    # Check for similar existing skills
    similar = self.retrieve(new_skill.description, top_k=3)

    for existing_skill in similar:
        similarity = cosine_similarity(new_skill.embedding, existing_skill.embedding)

        if similarity > threshold:
            # Merge or skip
            if new_skill.success_count > existing_skill.success_count:
                # Replace with better version
                self.skills[existing_skill.name] = new_skill
            return

    # Add as new skill
    self.add(new_skill)
```

### 2. Skill Composition

**Challenge:** How to combine multiple skills?

**Solution:** Dependency tracking
```python
def compose_skills(self, skill_names):
    """Generate code that composes multiple skills."""
    skills = [self.get_skill(name) for name in skill_names]

    # Sort by dependencies (topological sort)
    ordered = topological_sort(skills)

    # Combine code
    combined_code = "\n\n".join(skill.code for skill in ordered)

    return combined_code
```

### 3. Skill Pruning

**Problem:** Library grows too large.

**Solution:** Remove low-value skills
```python
def prune_library(self, max_size=100):
    """Keep only most valuable skills."""
    # Score each skill
    scored = []
    for skill in self.skills.values():
        # Value = usage * success_rate * recency
        usage = skill.success_count + skill.failure_count
        success_rate = skill.success_count / max(1, usage)
        recency = days_since(skill.created_at)

        score = usage * success_rate / (1 + recency)
        scored.append((score, skill))

    # Keep top skills
    scored.sort(reverse=True)
    self.skills = {s.name: s for score, s in scored[:max_size]}
```

---

## Pseudocode for Full Integration

```python
def train_with_skill_library(red_agent, blue_agent, tasks, num_iterations=10):
    """
    Co-evolutionary training with skill libraries.
    """
    adversarial_library = AdversarialSkillLibrary()

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}")

        for task in tasks:
            # Battle
            red_wins = adversarial_library.battle(task, red_agent, blue_agent)

            # Both agents learn from the battle
            if red_wins:
                # Blue team studies the attack
                attack_code = adversarial_library.battles[-1]['attack']
                blue_agent.learn_from_failure(task, attack_code)
            else:
                # Red team studies the defense
                defense_code = adversarial_library.battles[-1]['defense']
                red_agent.learn_from_failure(task, defense_code)

        # Periodic evaluation
        red_score = evaluate(red_agent, test_tasks)
        blue_score = evaluate(blue_agent, test_tasks)

        print(f"Red: {red_score}, Blue: {blue_score}")

    return adversarial_library
```

---

## Metrics for Evaluation

### Library Quality

```python
def evaluate_library_quality(skill_library):
    """Measure quality of skill library."""
    metrics = {}

    # Size
    metrics['num_skills'] = len(skill_library.skills)

    # Success rate
    total_success = sum(s.success_count for s in skill_library.skills.values())
    total_attempts = sum(
        s.success_count + s.failure_count
        for s in skill_library.skills.values()
    )
    metrics['avg_success_rate'] = total_success / max(1, total_attempts)

    # Diversity (average pairwise distance)
    embeddings = [s.embedding for s in skill_library.skills.values()]
    distances = [
        cosine_distance(e1, e2)
        for i, e1 in enumerate(embeddings)
        for e2 in embeddings[i+1:]
    ]
    metrics['diversity'] = np.mean(distances)

    return metrics
```

---

## Conclusion

Voyager provides the **skill accumulation mechanism** for our co-evolutionary framework. It proves that:

1. Code storage > Natural language storage (executable)
2. Retrieval + Composition enables complex behaviors
3. Automatic curriculum drives exploration
4. Skills transfer across environments

**For COEVOLVE:** This is the "tool library" that allows agents to build on past successes and compose solutions from reusable components.

---

## Next Steps

- [ ] Implement core skill library with vector search
- [ ] Implement code generation agent
- [ ] Implement automatic curriculum
- [ ] Test with code security scenarios
- [ ] Measure skill reuse and composition

---

## References

- Original Paper: https://arxiv.org/abs/2305.16291
- Project Page: https://voyager.minedojo.org/
- Code: https://github.com/MineDojo/Voyager
- Follow-up: "Ghost in the Minecraft" (similar work)
