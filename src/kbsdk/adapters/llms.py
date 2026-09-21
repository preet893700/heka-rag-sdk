"""LLM adapters (LangChain chat models behind the SDK's own `LLM` interface) and `ResilientLLM`."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar

from kbsdk.adapters._deps import Settings, api_key_from_env, require
from kbsdk.adapters.caches import cache_key
from kbsdk.errors import ConfigError
from kbsdk.interfaces import LLM, Cache
from kbsdk.resilience import AsyncRateLimiter, with_retries
from kbsdk.types import LLMResponse, Message, Usage


def _content_to_text(content: Any) -> str:
    """LangChain message content is a string or a list of blocks; keep only the visible text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


class ChatSettings(Settings):
    model: str
    api_key_env: str
    temperature: float = 0.0
    max_output_tokens: int | None = None
    timeout_s: float | None = 120.0


class GeminiSettings(ChatSettings):
    api_key_env: str = "GOOGLE_API_KEY"


class GroqSettings(ChatSettings):
    api_key_env: str = "GROQ_API_KEY"


class LangChainLLM:
    """Adapts any LangChain chat model to the SDK's `LLM` protocol."""

    _token_argument: ClassVar[str] = "max_output_tokens"

    def __init__(self, model: Any, name: str) -> None:
        self._model = model
        self.name = name

    def _bound(self, temperature: float | None, max_output_tokens: int | None) -> Any:
        overrides: dict[str, Any] = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_output_tokens is not None:
            overrides[self._token_argument] = max_output_tokens
        return self._model.bind(**overrides) if overrides else self._model

    @staticmethod
    def _convert(messages: Sequence[Message], system: str | None) -> list[Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        converted: list[Any] = [SystemMessage(content=system)] if system else []
        for message in messages:
            cls = HumanMessage if message.role == "user" else AIMessage
            converted.append(cls(content=message.content))
        return converted

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        model = self._bound(temperature, max_output_tokens)
        reply = await model.ainvoke(self._convert(messages, system))
        usage = getattr(reply, "usage_metadata", None) or {}
        return LLMResponse(
            text=_content_to_text(reply.content),
            usage=Usage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                llm_calls=1,
            ),
            model=self.name,
            raw=reply,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        model = self._bound(temperature, max_output_tokens)
        async for piece in model.astream(self._convert(messages, system)):
            text = _content_to_text(piece.content)
            if text:
                yield text


class GeminiLLM(LangChainLLM):
    settings_model = GeminiSettings

    def __init__(self, settings: GeminiSettings) -> None:
        module = require("langchain_google_genai", stage="llm", provider="gemini", extra="gemini")
        model = module.ChatGoogleGenerativeAI(
            model=settings.model,
            google_api_key=api_key_from_env(settings.api_key_env, "gemini"),
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            timeout=settings.timeout_s,
            max_retries=0,  # retries are handled once, by ResilientLLM
        )
        super().__init__(model, f"gemini:{settings.model}")


class GroqLLM(LangChainLLM):
    settings_model = GroqSettings
    _token_argument = "max_tokens"

    def __init__(self, settings: GroqSettings) -> None:
        module = require("langchain_groq", stage="llm", provider="groq", extra="groq")
        model = module.ChatGroq(
            model=settings.model,
            api_key=api_key_from_env(settings.api_key_env, "groq"),
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
            timeout=settings.timeout_s,
            max_retries=0,
        )
        super().__init__(model, f"groq:{settings.model}")


class ClaudeSettings(Settings):
    """No `temperature`: current Claude models reject sampling parameters (HTTP 400), so it is not a
    setting here and the SDK never sends one. Depth of reasoning is governed by the model itself."""

    model: str
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_output_tokens: int = 2048  # the Anthropic API requires an explicit ceiling
    timeout_s: float | None = 120.0


class ClaudeLLM(LangChainLLM):
    settings_model = ClaudeSettings
    _token_argument = "max_tokens"

    def __init__(self, settings: ClaudeSettings) -> None:
        module = require("langchain_anthropic", stage="llm", provider="claude", extra="anthropic")
        model = module.ChatAnthropic(
            model=settings.model,
            api_key=api_key_from_env(settings.api_key_env, "claude"),
            max_tokens=settings.max_output_tokens,
            timeout=settings.timeout_s,
            max_retries=0,
        )
        super().__init__(model, f"claude:{settings.model}")


class OpenAISettings(Settings):
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    # Some OpenAI models (reasoning models) reject a temperature: set `temperature: null` for those.
    temperature: float | None = 0.0
    max_output_tokens: int | None = None
    timeout_s: float | None = 120.0
    base_url: str | None = None  # any OpenAI-compatible endpoint (Azure, vLLM, a gateway)


class OpenAILLM(LangChainLLM):
    settings_model = OpenAISettings
    _token_argument = "max_tokens"

    def __init__(self, settings: OpenAISettings) -> None:
        module = require("langchain_openai", stage="llm", provider="openai", extra="openai")
        model = module.ChatOpenAI(
            model=settings.model,
            api_key=api_key_from_env(settings.api_key_env, "openai"),
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
            timeout=settings.timeout_s,
            base_url=settings.base_url,
            max_retries=0,
        )
        super().__init__(model, f"openai:{settings.model}")


class OllamaSettings(Settings):
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float | None = 0.0
    max_output_tokens: int | None = None
    timeout_s: float | None = 300.0  # local models can be slow on CPU


class OllamaLLM(LangChainLLM):
    """Local models through an Ollama server: no data leaves the machine."""

    settings_model = OllamaSettings
    _token_argument = "num_predict"

    def __init__(self, settings: OllamaSettings) -> None:
        module = require("langchain_ollama", stage="llm", provider="ollama", extra="ollama")
        model = module.ChatOllama(
            model=settings.model,
            base_url=settings.base_url,
            temperature=settings.temperature,
            num_predict=settings.max_output_tokens,
            client_kwargs={"timeout": settings.timeout_s},
        )
        super().__init__(model, f"ollama:{settings.model}")


class ResilientLLM:
    """Wraps an LLM with response caching, a shared rate limiter, retries and fallback models.

    Errors that retrying cannot fix (bad configuration) are raised immediately; anything else moves
    on to the next fallback model once retries for the current one are exhausted.
    """

    def __init__(
        self,
        inner: LLM,
        *,
        cache: Cache | None = None,
        limiter: AsyncRateLimiter | None = None,
        max_retries: int = 3,
        fallbacks: Sequence[LLM] = (),
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.cache = cache
        self.limiter = limiter or AsyncRateLimiter(None)
        self.max_retries = max_retries
        self.fallbacks = list(fallbacks)

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        key = cache_key(
            "llm",
            self.name,
            system,
            [m.model_dump() for m in messages],
            temperature,
            max_output_tokens,
        )
        if self.cache is not None:
            cached = await self.cache.get(key)
            if cached is not None:
                return LLMResponse(
                    text=cached["text"], usage=Usage(cache_hits=1), model=cached.get("model")
                )

        failure: Exception | None = None
        for llm in [self.inner, *self.fallbacks]:

            async def attempt(target: LLM = llm) -> LLMResponse:  # bound per loop iteration
                return await target.generate(
                    messages,
                    system=system,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )

            try:
                response = await with_retries(
                    attempt, max_retries=self.max_retries, before_attempt=self.limiter.wait
                )
            except ConfigError:
                raise
            except Exception as exc:
                failure = exc
                continue
            if self.cache is not None:
                await self.cache.set(key, {"text": response.text, "model": response.model})
            return response
        assert failure is not None
        raise failure

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        await self.limiter.wait()
        async for piece in self.inner.stream(
            messages, system=system, temperature=temperature, max_output_tokens=max_output_tokens
        ):
            yield piece
