# Generative Agents: Interactive Simulacra of Human Behavior

**Authors:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein
**Institution:** Stanford University, Google Research
**Year:** 2023
**Paper:** [arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)

---

## Core Idea

**Problem:** LLMs are stateless. They don't remember past interactions or form long-term goals.

**Solution:** Give agents a memory architecture that stores, retrieves, and reflects on experiences.

**Key Innovation:** Agents that behave believably over long time horizons by maintaining coherent memory and goals.

---

## The Memory Architecture

### Three Components

```
┌─────────────────────────────────────┐
│         MEMORY STREAM               │
│  (Chronological observations)       │
│                                     │
│  [t=1] John woke up                │
│  [t=2] John made coffee            │
│  [t=3] John talked to Maria        │
│  ...                               │
└─────────────────────────────────────┘
           │
           ├──► RETRIEVAL
           │    (Recency + Importance + Relevance)
           │
           └──► REFLECTION
                (High-level summaries)
```

### 1. Memory Stream

**What:** A database of all observations in chronological order.

**Format:**
```python
class Memory:
    timestamp: int
    description: str
    importance: float  # 1-10 scale
    embedding: np.array  # For similarity search
```

**Example:**
```
[t=08:00] John woke up feeling refreshed. (importance: 3)
[t=08:15] John made coffee. (importance: 2)
[t=09:00] John had a deep conversation with Maria about life goals. (importance: 8)
[t=10:00] John started working on his research project. (importance: 7)
```

### 2. Retrieval

**What:** How agents decide which memories to use when making decisions.

**Three Factors:**

**Recency:** Recent memories are more accessible
```python
recency_score = exp(-hours_since_memory)
```

**Importance:** Important memories are more salient
```python
importance = llm.rate(memory, scale=1-10)
```

**Relevance:** Memories related to current context
```python
relevance = cosine_similarity(query_embedding, memory_embedding)
```

**Combined Score:**
```python
retrieval_score = (
    α * recency_score +
    β * importance_score +
    γ * relevance_score
)
```

### 3. Reflection

**What:** Agents periodically generate high-level summaries of their memories.

**When:** After accumulating importance score > threshold (e.g., 100 points)

**Process:**
```
1. Retrieve recent high-importance memories
2. Ask LLM: "What are 3 high-level insights from these memories?"
3. Store insights as new memories (with high importance)
```

**Example:**

**Memories:**
```
- John had 5 conversations about career change
- John researched graduate programs
- John expressed dissatisfaction with current job
```

**Reflection:**
```
"John is seriously considering a career change and is researching options."
(importance: 9)
```

---

## The Planning Architecture

### Hierarchical Planning

**Long-term:** Daily plan
**Mid-term:** Hourly activities
**Short-term:** Minute-by-minute actions

**Example:**

```
Daily Plan:
  - Morning: Research project
  - Afternoon: Meet with Maria
  - Evening: Relax

Hourly Plan (for "Research project"):
  - Read 2 papers
  - Take notes
  - Draft outline

Action Plan (for "Read paper"):
  - Open paper on computer
  - Read abstract
  - Skim introduction
  - Take notes on key points
```

### Plan Generation

```python
def generate_plan(agent, timeframe="day"):
    # Retrieve relevant memories
    context = retrieve_memories(
        query=f"{agent.name}'s recent activities and goals",
        top_k=10
    )

    # Generate plan
    prompt = f"""
    Context: {context}
    Generate a {timeframe} plan for {agent.name}.
    """
    plan = llm.generate(prompt)

    return plan
```

---

## The Agent Loop

### High-Level Process

```python
while simulation_running:
    # 1. Perceive environment
    observations = agent.perceive(environment)

    # 2. Store observations in memory
    for obs in observations:
        importance = llm.rate_importance(obs)
        agent.memory.add(obs, importance)

    # 3. Reflect if threshold reached
    if agent.memory.total_importance > REFLECTION_THRESHOLD:
        insights = agent.reflect()
        agent.memory.add(insights, importance=9)

    # 4. Decide action
    context = agent.retrieve_relevant_memories()
    action = agent.plan_action(context)

    # 5. Execute action
    environment.execute(action)

    # 6. Update state
    agent.update_state()
```

---

## Experimental Results (from paper)

### Believability Study

**Setup:** Humans interact with 25 agents in a virtual town for 2 days.

**Evaluation:** Humans rate agent behavior on believability (1-10).

**Results:**

| Agent Type | Believability Score |
|------------|---------------------|
| Full Architecture | 8.2 |
| No Reflection | 6.4 |
| No Retrieval (all memories) | 5.1 |
| No Memory (stateless) | 3.9 |

**Conclusion:** All three components (memory, retrieval, reflection) are critical.

### Emergent Social Behaviors

**Observed Behaviors:**
- Information diffusion (gossip spreads through network)
- Coordination (agents spontaneously organize a party)
- Relationship formation (agents form friendships based on interactions)

**Key Finding:** Simple memory + retrieval + reflection → Complex social dynamics

---

## Relation to COEVOLVE

### How We Use It

1. **Shared Genetic Memory**
   - All agents access a shared memory pool
   - Successful strategies remembered
   - Failures forgotten (or marked as "anti-patterns")

2. **Debate History**
   - Store all debates in memory
   - Retrieve similar past debates
   - Learn from historical arguments

3. **Skill Library** (like Voyager)
   - Store successful code/reasoning patterns
   - Retrieve for similar problems
   - Reflect on meta-patterns

### Novel Extensions

**Competitive Memory:**
- Agent A stores "how to attack Agent B"
- Agent B stores "how to defend against Agent A"
- Co-evolution through memory

**Meta-Reflection:**
- Reflect on reflection quality
- "Which types of insights were most useful?"
- Learn what to remember

**Memory Pruning:**
- Biological forgetting
- Remove low-importance, old memories
- Prevent memory bloat

---

## Integration with Other Papers

**With Debate:**
- Debates stored in memory
- Retrieve similar debates for new arguments
- Reflect on debate strategies

**With Constitutional AI:**
- Constitution stored in memory
- Retrieve relevant principles for current task
- Reflect on constitutional compliance

**With STaR:**
- Successful reasoning chains stored
- Retrieve for similar problems
- Reflect on reasoning patterns

**With Voyager:**
- Skills = specialized memories
- Retrieval by task similarity
- Reflection on skill effectiveness

---

## Implementation for COEVOLVE

### Core Memory System

```python
from dataclasses import dataclass
from typing import List
import numpy as np
from datetime import datetime

@dataclass
class Memory:
    """A single memory entry."""
    timestamp: datetime
    description: str
    importance: float  # 1-10
    embedding: np.array
    memory_type: str  # "observation", "reflection", "plan"
    related_memories: List[int] = None  # IDs of related memories

class MemoryStream:
    """Chronological memory storage."""

    def __init__(self, embedding_model):
        self.memories = []
        self.embedding_model = embedding_model
        self.reflection_threshold = 100
        self.total_importance = 0

    def add(self, description, importance, memory_type="observation"):
        """Add a new memory."""
        embedding = self.embedding_model.encode(description)

        memory = Memory(
            timestamp=datetime.now(),
            description=description,
            importance=importance,
            embedding=embedding,
            memory_type=memory_type
        )

        self.memories.append(memory)
        self.total_importance += importance

        # Check if reflection needed
        if self.total_importance > self.reflection_threshold:
            self.trigger_reflection()

        return memory

    def retrieve(self, query, top_k=5, alpha=1.0, beta=1.0, gamma=1.0):
        """Retrieve relevant memories."""
        query_embedding = self.embedding_model.encode(query)
        now = datetime.now()

        scored_memories = []

        for memory in self.memories:
            # Recency (decay over time)
            hours_ago = (now - memory.timestamp).total_seconds() / 3600
            recency = np.exp(-0.01 * hours_ago)

            # Importance (normalized)
            importance = memory.importance / 10.0

            # Relevance (cosine similarity)
            relevance = np.dot(query_embedding, memory.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding)
            )

            # Combined score
            score = alpha * recency + beta * importance + gamma * relevance
            scored_memories.append((score, memory))

        # Sort and return top-k
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [mem for score, mem in scored_memories[:top_k]]

    def trigger_reflection(self):
        """Generate high-level insights."""
        # Retrieve recent important memories
        recent_important = [
            m for m in self.memories[-100:]
            if m.importance >= 7
        ]

        if len(recent_important) < 3:
            return

        # Ask LLM for insights
        memory_descriptions = [m.description for m in recent_important]
        insights = self.generate_insights(memory_descriptions)

        # Store insights as new memories
        for insight in insights:
            self.add(insight, importance=8, memory_type="reflection")

        # Reset importance counter
        self.total_importance = 0

    def generate_insights(self, memories):
        """Use LLM to generate insights from memories."""
        prompt = f"""
        Given these recent important memories:
        {chr(10).join(f"- {m}" for m in memories)}

        Generate 3 high-level insights or patterns.
        """
        # Call LLM here
        insights = llm.generate(prompt)
        return insights.split("\n")
```

### Agent with Memory

```python
class MemoryAgent:
    """Agent with generative memory architecture."""

    def __init__(self, name, model, embedding_model):
        self.name = name
        self.model = model
        self.memory = MemoryStream(embedding_model)
        self.current_plan = None

    def observe(self, observation, importance=None):
        """Perceive and store observation."""
        if importance is None:
            # Ask LLM to rate importance
            importance = self.rate_importance(observation)

        self.memory.add(observation, importance, memory_type="observation")

    def rate_importance(self, observation):
        """Rate importance of observation (1-10)."""
        prompt = f"""
        On a scale of 1-10, how important is this observation for {self.name}?
        Observation: {observation}

        Return only a number 1-10.
        """
        response = self.model.generate(prompt)
        try:
            return float(response.strip())
        except:
            return 5.0  # Default

    def plan_action(self, goal):
        """Plan action based on goal and memories."""
        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve(goal, top_k=10)

        # Generate plan
        context = "\n".join(m.description for m in relevant_memories)
        prompt = f"""
        Context (relevant memories):
        {context}

        Goal: {goal}

        Plan the next action for {self.name}:
        """
        action = self.model.generate(prompt)
        return action

    def reflect(self):
        """Manually trigger reflection."""
        self.memory.trigger_reflection()
```

---

## Novel: Multi-Agent Shared Memory

```python
class SharedMemoryPool:
    """Memory pool shared across multiple agents."""

    def __init__(self, embedding_model):
        self.memory_stream = MemoryStream(embedding_model)
        self.agent_memories = {}  # Per-agent private memories

    def add_shared(self, description, importance, contributor):
        """Add to shared pool (visible to all)."""
        memory = self.memory_stream.add(description, importance)
        memory.contributor = contributor
        return memory

    def add_private(self, agent_id, description, importance):
        """Add to agent's private memory."""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = MemoryStream(self.embedding_model)

        return self.agent_memories[agent_id].add(description, importance)

    def retrieve_for_agent(self, agent_id, query, include_shared=True):
        """Retrieve memories for a specific agent."""
        all_memories = []

        # Private memories
        if agent_id in self.agent_memories:
            private = self.agent_memories[agent_id].retrieve(query)
            all_memories.extend(private)

        # Shared memories
        if include_shared:
            shared = self.memory_stream.retrieve(query)
            all_memories.extend(shared)

        # Re-rank combined memories
        # ... (implement re-ranking logic)

        return all_memories
```

---

## Research Questions for COEVOLVE

1. **Optimal memory architecture for co-evolution?**
   - Fully shared vs. Partially shared vs. Private
   - Measure: Information transfer vs. diversity

2. **Reflection frequency trade-off?**
   - More reflection = better insights but slower
   - Measure: Quality vs. compute cost

3. **What should agents remember from debates?**
   - Arguments? Outcomes? Opponent strategies?
   - Measure: Future debate performance

4. **Memory pruning strategies?**
   - Age-based vs. Importance-based vs. Relevance-based
   - Measure: Performance vs. memory size

---

## Critical Implementation Details

### 1. Embedding Model

**Options:**
```python
# Option 1: Sentence transformers (lightweight)
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Option 2: OpenAI embeddings (more powerful)
import openai
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']
```

### 2. Reflection Triggers

**Simple:** Fixed importance threshold
```python
if total_importance > 100:
    reflect()
```

**Smart:** Dynamic threshold
```python
if total_importance > base_threshold * (1 + num_reflections * 0.1):
    reflect()
```

**Periodic:** Time-based
```python
if hours_since_last_reflection > 4:
    reflect()
```

### 3. Memory Pruning

```python
def prune_memories(self, max_size=1000):
    """Remove low-value old memories."""
    if len(self.memories) <= max_size:
        return

    # Score each memory
    now = datetime.now()
    scored = []

    for memory in self.memories:
        # Age penalty
        hours_old = (now - memory.timestamp).total_seconds() / 3600
        age_factor = np.exp(-0.001 * hours_old)

        # Importance boost
        importance_factor = memory.importance / 10

        # Keep reflections longer
        type_boost = 2.0 if memory.memory_type == "reflection" else 1.0

        score = importance_factor * age_factor * type_boost
        scored.append((score, memory))

    # Keep top memories
    scored.sort(reverse=True)
    self.memories = [mem for score, mem in scored[:max_size]]
```

---

## Metrics for Evaluation

### Memory Quality

```python
def evaluate_memory_quality(agent, test_queries):
    """Measure retrieval quality."""
    scores = []

    for query, ground_truth_memory in test_queries:
        retrieved = agent.memory.retrieve(query, top_k=5)

        # Check if ground truth is in top-5
        found = ground_truth_memory in [m.description for m in retrieved]
        scores.append(1.0 if found else 0.0)

    return np.mean(scores)
```

### Reflection Quality

```python
def evaluate_reflection_quality(agent):
    """Measure quality of generated insights."""
    reflections = [
        m for m in agent.memory.memories
        if m.memory_type == "reflection"
    ]

    # Check if reflections are useful for decision-making
    # (This requires domain-specific evaluation)

    return len(reflections), avg_usefulness_score
```

---

## Conclusion

Generative Agents provides the **memory architecture** for our co-evolutionary framework. It proves that:

1. Long-term memory enables coherent behavior
2. Retrieval by recency + importance + relevance works well
3. Reflection creates higher-level knowledge
4. Memory enables emergent social behaviors

**For COEVOLVE:** This is the "genetic memory" that allows agents to accumulate and transfer knowledge across generations.

---

## Next Steps

- [ ] Implement core memory stream
- [ ] Implement retrieval with three factors
- [ ] Implement reflection mechanism
- [ ] Test with debate scenario
- [ ] Measure memory quality and retrieval accuracy

---

## References

- Original Paper: https://arxiv.org/abs/2304.03442
- Interactive Demo: https://reverie.herokuapp.com/arXiv_Demo/
- GitHub: https://github.com/joonspk-research/generative_agents
