"""Adapter wiring without any network: construction, key handling, response parsing, resilience."""

import pytest

from conftest import ScriptedLLM
from kbsdk import ConfigError, Message, RAGConfig, registry
from kbsdk.adapters.caches import MemoryCache
from kbsdk.adapters.llms import LangChainLLM, ResilientLLM, _content_to_text
from kbsdk.factory import build_llm
from kbsdk.interfaces import LLM
from kbsdk.types import LLMResponse, Usage

pytest.importorskip("langchain_core")


def test_content_blocks_reduce_to_visible_text():
    assert _content_to_text("plain") == "plain"
    blocks = [
        {"type": "thinking", "thinking": "hidden"},
        {"type": "text", "text": "Hello "},
        "world",
    ]
    assert _content_to_text(blocks) == "Hello world"
    assert _content_to_text(None) == ""


@pytest.mark.parametrize(
    ("provider", "env"), [("gemini", "GOOGLE_API_KEY"), ("groq", "GROQ_API_KEY")]
)
def test_providers_construct_offline_and_conform(monkeypatch, provider, env):
    monkeypatch.setenv(env, "dummy-key-not-real")
    llm = registry.create(
        "llm", provider, {"model": "some-model", "temperature": 0.1, "max_output_tokens": 50}
    )
    assert isinstance(llm, LLM)
    assert llm.name == f"{provider}:some-model"


@pytest.mark.parametrize(
    ("provider", "env"), [("gemini", "GOOGLE_API_KEY"), ("groq", "GROQ_API_KEY")]
)
def test_missing_key_gives_a_clear_error(monkeypatch, provider, env):
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(ConfigError, match=env):
        registry.create("llm", provider, {"model": "m"})


def test_typos_in_llm_params_are_rejected(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    with pytest.raises(ValueError):
        registry.create("llm", "gemini", {"model": "m", "temprature": 0.2})


def test_api_key_env_can_name_any_variable(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("MY_TEAM_KEY", "dummy")
    assert registry.create("llm", "gemini", {"model": "m", "api_key_env": "MY_TEAM_KEY"})


async def test_langchain_wrapper_converts_messages_and_usage():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    llm = LangChainLLM(FakeListChatModel(responses=["hello there"]), "fake:lc")
    response = await llm.generate([Message(role="user", content="hi")], system="be brief")
    assert response.text == "hello there"
    assert response.model == "fake:lc"
    pieces = [
        p
        async for p in LangChainLLM(FakeListChatModel(responses=["abc"]), "x").stream(
            [Message(role="user", content="hi")]
        )
    ]
    assert "".join(pieces) == "abc"


class Sequenced:
    """Fails with the given errors in order, then answers."""

    name = "seq"

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    async def generate(self, messages, *, system=None, temperature=None, max_output_tokens=None):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return LLMResponse(
            text="ok", usage=Usage(input_tokens=3, output_tokens=1, llm_calls=1), model=self.name
        )

    async def stream(self, *a, **k):
        yield "ok"


def rate_limit():
    error = RuntimeError("quota")
    error.status_code = 429
    return error


async def test_resilient_llm_retries_and_caches(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("kbsdk.resilience.asyncio.sleep", instant)
    inner = Sequenced(rate_limit(), rate_limit())
    llm = ResilientLLM(inner, cache=MemoryCache(), max_retries=3)
    messages = [Message(role="user", content="q")]
    first = await llm.generate(messages, system="s")
    assert first.text == "ok" and inner.calls == 3
    second = await llm.generate(messages, system="s")
    assert second.text == "ok" and inner.calls == 3  # served from cache, no new call
    assert second.usage.cache_hits == 1 and second.usage.input_tokens == 0
    await llm.generate(messages, system="different")
    assert inner.calls == 4  # a different prompt is a different cache entry


async def test_resilient_llm_falls_back_after_exhausting_retries(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("kbsdk.resilience.asyncio.sleep", instant)
    primary = Sequenced(*[rate_limit() for _ in range(10)])
    backup = ScriptedLLM(lambda m, s: "from backup", name="backup")
    llm = ResilientLLM(primary, max_retries=1, fallbacks=[backup])
    response = await llm.generate([Message(role="user", content="q")])
    assert response.text == "from backup" and primary.calls == 2


async def test_resilient_llm_raises_when_everything_fails(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("kbsdk.resilience.asyncio.sleep", instant)
    llm = ResilientLLM(Sequenced(*[rate_limit() for _ in range(5)]), max_retries=1)
    with pytest.raises(RuntimeError, match="quota"):
        await llm.generate([Message(role="user", content="q")])


async def test_config_errors_are_never_retried_or_masked_by_fallbacks():
    boom = ConfigError("bad key")
    inner = Sequenced(boom)
    llm = ResilientLLM(inner, max_retries=3, fallbacks=[ScriptedLLM()])
    with pytest.raises(ConfigError):
        await llm.generate([Message(role="user", content="q")])
    assert inner.calls == 1


def test_defaults_are_only_passed_to_providers_that_declare_them(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    config = RAGConfig.from_dict(
        {"generation": {"llm": {"provider": "gemini", "params": {"model": "m"}}}}
    )
    llm = build_llm(
        config, config.generation.llm, None, defaults={"temperature": 0.3, "not_a_setting": 1}
    )
    assert llm.name == "gemini:m"  # 'not_a_setting' was filtered out instead of failing validation

    class Bare:  # a plug-in with no settings model at all
        name = "bare"

        def __init__(self):
            pass

    registry.register("llm", "bare_plugin")(Bare)
    plain = RAGConfig.from_dict({"generation": {"llm": {"provider": "bare_plugin"}}})
    assert (
        build_llm(plain, plain.generation.llm, None, defaults={"temperature": 0.3}).name == "bare"
    )


def test_build_llm_applies_reliability_settings(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    config = RAGConfig.preset("free-tier-dev")
    llm = build_llm(config, config.generation.llm, MemoryCache(), defaults={"temperature": 0.0})
    assert isinstance(llm, ResilientLLM)
    assert llm.max_retries == 5
    assert llm.limiter.requests_per_minute == 8
    assert llm.cache is not None


def test_gemini_afc_notice_is_hidden_but_other_warnings_from_that_logger_are_not(caplog):
    import logging

    from kbsdk.adapters.llms import silence_gemini_afc_notice

    silence_gemini_afc_notice()
    silence_gemini_afc_notice()  # idempotent: one filter, not two
    genai_logger = logging.getLogger("google_genai.models")
    assert len(genai_logger.filters) == 1
    with caplog.at_level(logging.WARNING, logger="google_genai.models"):
        genai_logger.warning(
            "Direct use of automatic function calling (AFC) in AsyncModels.generate_content is "
            "not recommended."
        )
        genai_logger.warning("quota is nearly exhausted")
    assert (
        "quota is nearly exhausted" in caplog.text
        and "automatic function calling" not in caplog.text
    )
