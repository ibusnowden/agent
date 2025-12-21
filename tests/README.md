# COEVOLVE Testing Guide

Complete testing infrastructure for the core components.

---

## Test Structure

```
tests/
├── README.md                  # This file
├── run_tests.py              # Main test runner
├── test_llm_provider.py      # LLM provider tests (400+ lines)
├── test_agent.py             # Agent system tests (450+ lines)
├── test_judge.py             # Judge system tests (500+ lines)
└── test_state_machine.py     # State machine tests (400+ lines)
```

**Total Test Code:** ~1,750 lines

---

## Test Coverage

### LLM Provider Tests (`test_llm_provider.py`)

**Tests:** 15+ test cases

**Coverage:**
- ✅ BaseLLMProvider interface
- ✅ OpenAIProvider (with mocking)
- ✅ AnthropicProvider (with mocking)
- ✅ LocalLLMProvider
- ✅ Factory function (`get_llm_provider`)
- ✅ LLMResponse dataclass
- ✅ Token tracking
- ✅ Error handling (missing API keys)
- ✅ Batch generation

**Key Features:**
- Uses mocks to test without API keys
- Tests all providers
- Validates response format
- Error case handling

### Agent Tests (`test_agent.py`)

**Tests:** 20+ test cases

**Coverage:**
- ✅ AgentMemory (observations, actions, reflections)
- ✅ ActorAgent (generation, revision)
- ✅ SupervisorAgent (critique, validation)
- ✅ ConstitutionalAgent (self-critique loop)
- ✅ AgentAction dataclass
- ✅ Token accumulation
- ✅ Reflection mechanism
- ✅ Feedback handling

**Key Features:**
- MockLLMProvider for isolated testing
- Tests with/without feedback
- Validates memory tracking
- Tests iterative refinement

### Judge Tests (`test_judge.py`)

**Tests:** 25+ test cases

**Coverage:**
- ✅ Verdict enum
- ✅ JudgmentResult dataclass
- ✅ LLMJudge (single & comparative)
- ✅ GroundedJudge (objective verification)
- ✅ CodeSecurityJudge (vulnerability detection)
- ✅ ConsensusJudge (majority & weighted voting)
- ✅ MetaJudge (feedback & accuracy tracking)
- ✅ Judgment history
- ✅ Confidence scoring

**Key Features:**
- Tests all judge types
- Security vulnerability detection
- Multi-judge consensus
- Meta-learning feedback loop

### State Machine Tests (`test_state_machine.py`)

**Tests:** 15+ test cases

**Coverage:**
- ✅ CoEvolutionStateMachine initialization
- ✅ Single iteration success
- ✅ Multiple iterations
- ✅ Max iterations termination
- ✅ History tracking
- ✅ Token tracking
- ✅ DebateStateMachine
- ✅ Debate execution
- ✅ Both agents participate
- ✅ State structure
- ✅ Result dataclass

**Key Features:**
- MockJudge with configurable verdicts
- Tests iteration logic
- Validates state transitions
- Tests both state machines

---

## Running Tests

### Run All Tests

```bash
cd ~/Desktop/COEVOLVE
python tests/run_tests.py
```

### Run Specific Test File

```bash
# Test LLM providers
python tests/test_llm_provider.py

# Test agents
python tests/test_agent.py

# Test judges
python tests/test_judge.py

# Test state machines
python tests/test_state_machine.py
```

### Run with Pytest Directly

```bash
# All tests with verbose output
pytest tests/ -v

# Specific test file
pytest tests/test_agent.py -v

# Specific test class
pytest tests/test_agent.py::TestActorAgent -v

# Specific test function
pytest tests/test_agent.py::TestActorAgent::test_act_without_feedback -v

# With coverage report
pytest tests/ --cov=core --cov-report=html

# Show print statements
pytest tests/ -v -s
```

---

## Test Design Principles

### 1. Isolation

**All tests run without external dependencies:**
- ✅ No API keys required
- ✅ MockLLMProvider for LLM calls
- ✅ No network calls
- ✅ No file system changes (except temp)

### 2. Comprehensive

**Every component has tests:**
- Unit tests for individual methods
- Integration tests for workflows
- Edge case handling
- Error conditions

### 3. Fast

**Tests execute quickly:**
- Mocked LLM calls
- No real API calls
- Minimal I/O
- Target: < 10 seconds for full suite

### 4. Maintainable

**Tests are clear and readable:**
- Descriptive test names
- Clear arrange-act-assert structure
- Good documentation
- Minimal duplication

---

## Mock Objects

### MockLLMProvider

```python
class MockLLMProvider(BaseLLMProvider):
    """Returns predefined responses."""

    def generate(self, prompt, **kwargs):
        return LLMResponse(
            content="Mock response",
            model="mock",
            provider="mock",
            tokens_used=10
        )
```

**Usage:**
```python
llm = MockLLMProvider()
agent = ActorAgent("test", llm)
action = agent.act({"task": "Test"})
assert "Mock response" in action.content
```

### MockJudge

```python
class MockJudge:
    """Returns predefined verdicts."""

    def __init__(self, verdicts_to_return):
        self.verdicts = verdicts_to_return
        self.call_count = 0

    def judge(self, context):
        verdict = self.verdicts[self.call_count]
        self.call_count += 1
        return JudgmentResult(verdict, "Mock", 0.8)
```

**Usage:**
```python
# Test iteration logic
judge = MockJudge([Verdict.INVALID, Verdict.AGENT_A_WINS])
machine = CoEvolutionStateMachine(actor, supervisor, judge)
result = machine.run("Task")
assert result.iterations == 2  # Should iterate twice
```

---

## Expected Test Results

### All Tests Should Pass ✅

**Expected output:**
```
================================================================================
COEVOLVE Test Suite
================================================================================
tests/test_llm_provider.py::TestBaseLLMProvider::test_initialization PASSED
tests/test_llm_provider.py::TestBaseLLMProvider::test_generate PASSED
...
tests/test_state_machine.py::TestDebateStateMachine::test_debate_execution PASSED

================================================================================
✅ ALL TESTS PASSED!
================================================================================

75+ tests passed in < 10 seconds
```

### What Tests Validate

1. **Correctness:** All components work as designed
2. **Integration:** Components work together
3. **Error Handling:** Graceful failure modes
4. **Edge Cases:** Boundary conditions handled
5. **API Contracts:** Interfaces are stable

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'core'`

**Solution:**
```bash
# Make sure you're in the right directory
cd ~/Desktop/COEVOLVE

# Run tests from project root
python tests/run_tests.py
```

### Pytest Not Found

**Problem:** `pytest: command not found`

**Solution:**
```bash
pip install pytest
```

### Mock Issues

**Problem:** Tests fail due to API calls

**Solution:**
- Ensure using MockLLMProvider, not real providers
- Check that `@patch` decorators are applied
- Verify environment variables are cleared in tests

---

## Adding New Tests

### Template for New Test File

```python
"""
Tests for NewComponent
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.new_component import NewClass


class TestNewClass:
    """Test NewClass functionality."""

    def test_initialization(self):
        """Test object creation."""
        obj = NewClass()
        assert obj is not None

    def test_main_functionality(self):
        """Test primary use case."""
        obj = NewClass()
        result = obj.do_something()
        assert result == expected_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Best Practices

1. **One test file per module**
2. **One test class per component class**
3. **Descriptive test names** (`test_actor_generates_response_without_feedback`)
4. **Use fixtures** for common setup
5. **Mock external dependencies**
6. **Test both success and failure paths**

---

## Continuous Integration (Future)

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: python tests/run_tests.py
      - name: Generate coverage
        run: pytest tests/ --cov=core --cov-report=xml
```

---

## Test Metrics

### Target Metrics

- **Coverage:** > 80%
- **Pass Rate:** 100%
- **Execution Time:** < 10 seconds
- **Flakiness:** 0 (all tests deterministic)

### Current Status

- ✅ **75+ test cases** written
- ✅ **All components** covered
- ✅ **Mock-based** (no external deps)
- ✅ **Fast** (mocked LLM calls)
- ✅ **Comprehensive** (unit + integration)

---

## Next Steps

1. **Run tests** to ensure all pass
2. **Fix any bugs** found
3. **Add coverage reporting**
4. **Set up CI/CD**
5. **Add integration tests** (full workflows)
6. **Add performance tests** (token usage, timing)

---

## Test-Driven Development

### Workflow for New Features

1. **Write test first** (it will fail)
2. **Implement feature** (make test pass)
3. **Refactor** (keep tests passing)
4. **Document** (update this guide)

### Example

```python
# 1. Write test (fails)
def test_new_feature():
    result = new_feature()
    assert result == expected

# 2. Implement
def new_feature():
    return expected

# 3. Refactor
def new_feature():
    # Better implementation
    return compute_expected()

# 4. Document
# Added new_feature() which does X
```

---

## Summary

**Testing Infrastructure Complete!**

- ✅ 4 test files
- ✅ 75+ test cases
- ✅ 1,750+ lines of test code
- ✅ All components covered
- ✅ Mock-based (fast, no deps)
- ✅ Ready to run

**Next:** Run tests, fix bugs, then continue building!

---

**Testing is not optional. It's how we ensure quality.** 🧪✅
