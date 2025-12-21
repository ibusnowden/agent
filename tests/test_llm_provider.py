"""
Unit Tests for LLM Provider System

Tests the provider abstraction layer without requiring API keys.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_provider import (
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    LocalLLMProvider,
    LLMResponse,
    get_llm_provider,
)


class MockLLMProvider(BaseLLMProvider):
    """Mock provider for testing."""

    def generate(self, prompt, system_prompt=None, **kwargs):
        return LLMResponse(
            content=f"Mock response to: {prompt[:50]}",
            model=self.model_name,
            provider="mock",
            tokens_used=10
        )

    def generate_batch(self, prompts, system_prompt=None, **kwargs):
        return [self.generate(p, system_prompt, **kwargs) for p in prompts]


class TestBaseLLMProvider:
    """Test the base provider interface."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = MockLLMProvider(
            model_name="test-model",
            temperature=0.5,
            max_tokens=100
        )

        assert provider.model_name == "test-model"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 100

    def test_generate(self):
        """Test single generation."""
        provider = MockLLMProvider("test-model")
        response = provider.generate("Test prompt")

        assert isinstance(response, LLMResponse)
        assert "Mock response" in response.content
        assert response.model == "test-model"
        assert response.provider == "mock"
        assert response.tokens_used == 10

    def test_generate_batch(self):
        """Test batch generation."""
        provider = MockLLMProvider("test-model")
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = provider.generate_batch(prompts)

        assert len(responses) == 3
        assert all(isinstance(r, LLMResponse) for r in responses)


class TestOpenAIProvider:
    """Test OpenAI provider (with mocking)."""

    @patch('core.llm_provider.openai.OpenAI')
    def test_initialization(self, mock_openai):
        """Test OpenAI provider initialization."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider(model_name="gpt-3.5-turbo")
            assert provider.model_name == "gpt-3.5-turbo"

    @patch('core.llm_provider.openai.OpenAI')
    def test_generate(self, mock_openai_class):
        """Test OpenAI generation with mock."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage.total_tokens = 50

        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider(model_name="gpt-3.5-turbo")
            response = provider.generate("Test prompt")

            assert response.content == "Test response"
            assert response.tokens_used == 50
            assert response.provider == "openai"

    def test_api_key_required(self):
        """Test that API key is required."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove OPENAI_API_KEY from environment
            if 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']

            with pytest.raises(ValueError, match="API key"):
                OpenAIProvider(model_name="gpt-3.5-turbo")


class TestAnthropicProvider:
    """Test Anthropic provider (with mocking)."""

    @patch('core.llm_provider.anthropic.Anthropic')
    def test_initialization(self, mock_anthropic):
        """Test Anthropic provider initialization."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = AnthropicProvider(model_name="claude-3-opus-20240229")
            assert provider.model_name == "claude-3-opus-20240229"

    @patch('core.llm_provider.anthropic.Anthropic')
    def test_generate(self, mock_anthropic_class):
        """Test Anthropic generation with mock."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        mock_response.model = "claude-3-opus-20240229"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 40

        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = AnthropicProvider(model_name="claude-3-opus-20240229")
            response = provider.generate("Test prompt")

            assert response.content == "Test response"
            assert response.tokens_used == 50  # input + output
            assert response.provider == "anthropic"


class TestLocalLLMProvider:
    """Test local LLM provider."""

    @patch('core.llm_provider.openai.OpenAI')
    def test_initialization(self, mock_openai):
        """Test local provider initialization."""
        provider = LocalLLMProvider(
            model_name="llama-2-7b",
            api_base="http://localhost:8000/v1"
        )
        assert provider.model_name == "llama-2-7b"


class TestFactoryFunction:
    """Test the get_llm_provider factory function."""

    @patch('core.llm_provider.openai.OpenAI')
    def test_get_openai_provider(self, mock_openai):
        """Test creating OpenAI provider via factory."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = get_llm_provider("openai", "gpt-4")
            assert isinstance(provider, OpenAIProvider)
            assert provider.model_name == "gpt-4"

    @patch('core.llm_provider.anthropic.Anthropic')
    def test_get_anthropic_provider(self, mock_anthropic):
        """Test creating Anthropic provider via factory."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = get_llm_provider("anthropic", "claude-3-opus-20240229")
            assert isinstance(provider, AnthropicProvider)

    @patch('core.llm_provider.openai.OpenAI')
    def test_get_local_provider(self, mock_openai):
        """Test creating local provider via factory."""
        provider = get_llm_provider("local", "llama-2-7b")
        assert isinstance(provider, LocalLLMProvider)

    def test_invalid_provider(self):
        """Test that invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm_provider("invalid", "model")


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_creation(self):
        """Test creating LLMResponse."""
        response = LLMResponse(
            content="Test content",
            model="test-model",
            provider="test",
            tokens_used=42
        )

        assert response.content == "Test content"
        assert response.model == "test-model"
        assert response.provider == "test"
        assert response.tokens_used == 42
        assert response.finish_reason == "stop"  # default


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
