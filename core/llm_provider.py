"""
LLM Provider Abstraction

Unified interface for different LLM providers (OpenAI, Anthropic, local models).
Enables easy switching between models and providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os
from enum import Enum

# Import providers (with graceful fallbacks)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    finish_reason: str = "stop"
    raw_response: Any = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (GPT-3.5, GPT-4, etc.)."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        # Set API key
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var.")

        self.client = openai.OpenAI(api_key=openai.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI Chat API."""

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )

        # Parse response
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider="openai",
            tokens_used=response.usage.total_tokens,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response
        )

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, system_prompt, **kwargs) for p in prompts]


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider (Claude models)."""

    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

        # Set API key
        if api_key:
            self.api_key = api_key
        elif os.getenv("ANTHROPIC_API_KEY"):
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY env var.")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Call API
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            system=system_prompt or "",
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )

        # Parse response
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider="anthropic",
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason,
            raw_response=response
        )

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, system_prompt, **kwargs) for p in prompts]


class LocalLLMProvider(BaseLLMProvider):
    """Local LLM provider (VLLM, llama.cpp, etc.)."""

    def __init__(
        self,
        model_name: str = "llama-2-7b-chat",
        api_base: str = "http://localhost:8000/v1",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package needed for local models. Run: pip install openai")

        # Use OpenAI-compatible API
        self.client = openai.OpenAI(
            api_key="dummy",  # Local servers don't need real keys
            base_url=api_base
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using local model."""

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call local API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )

        # Parse response
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model_name,
            provider="local",
            tokens_used=getattr(response.usage, 'total_tokens', 0),
            finish_reason=response.choices[0].finish_reason,
            raw_response=response
        )

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, system_prompt, **kwargs) for p in prompts]


def get_llm_provider(
    provider: str,
    model_name: str,
    **kwargs
) -> BaseLLMProvider:
    """
    Factory function to get LLM provider.

    Args:
        provider: "openai", "anthropic", or "local"
        model_name: Model identifier
        **kwargs: Additional configuration

    Returns:
        Configured LLM provider

    Example:
        >>> llm = get_llm_provider("openai", "gpt-4")
        >>> response = llm.generate("Hello, world!")
        >>> print(response.content)
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAIProvider(model_name=model_name, **kwargs)
    elif provider == "anthropic":
        return AnthropicProvider(model_name=model_name, **kwargs)
    elif provider == "local":
        return LocalLLMProvider(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Convenience functions

def create_llm_from_config(config):
    """Create LLM provider from configuration object."""
    from .config import ModelConfig

    if isinstance(config, ModelConfig):
        return get_llm_provider(
            provider=config.provider,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            api_base=config.api_base
        )
    else:
        raise TypeError("Expected ModelConfig object")


# Example usage
if __name__ == "__main__":
    # Example: OpenAI
    print("Testing OpenAI provider...")
    try:
        llm = get_llm_provider("openai", "gpt-3.5-turbo")
        response = llm.generate(
            "What is 2+2?",
            system_prompt="You are a helpful math assistant."
        )
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.tokens_used}")
    except Exception as e:
        print(f"OpenAI test failed: {e}")

    # Example: Anthropic
    print("\nTesting Anthropic provider...")
    try:
        llm = get_llm_provider("anthropic", "claude-3-sonnet-20240229")
        response = llm.generate(
            "What is the capital of France?",
            system_prompt="You are a helpful geography assistant."
        )
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.tokens_used}")
    except Exception as e:
        print(f"Anthropic test failed: {e}")

    # Example: Local
    print("\nTesting Local provider...")
    try:
        llm = get_llm_provider(
            "local",
            "llama-2-7b-chat",
            api_base="http://localhost:8000/v1"
        )
        response = llm.generate("Hello!")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Local test failed (is server running?): {e}")
