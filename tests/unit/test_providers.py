"""New providers: construction, key handling and wiring, all without a network or a real key."""

import pytest
from pydantic import ValidationError

from heka.rag import ConfigError, RAGConfig, registry
from heka.rag.config import available_presets
from heka.rag.factory import build_llm, generation_defaults
from heka.rag.interfaces import LLM, Embedder

pytest.importorskip("langchain_core")

CLOUD_LLMS = [
    ("claude", "ANTHROPIC_API_KEY", "claude-opus-5"),
    ("openai", "OPENAI_API_KEY", "some-model"),
]


@pytest.mark.parametrize(("provider", "env", "model"), CLOUD_LLMS)
def test_cloud_llms_construct_offline_and_conform(monkeypatch, provider, env, model):
    pytest.importorskip({"claude": "langchain_anthropic", "openai": "langchain_openai"}[provider])
    monkeypatch.setenv(env, "dummy-key-not-real")
    llm = registry.create("llm", provider, {"model": model})
    assert isinstance(llm, LLM) and llm.name == f"{provider}:{model}"


@pytest.mark.parametrize(("provider", "env", "model"), CLOUD_LLMS)
def test_a_missing_key_names_the_variable(monkeypatch, provider, env, model):
    pytest.importorskip({"claude": "langchain_anthropic", "openai": "langchain_openai"}[provider])
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(ConfigError, match=env):
        registry.create("llm", provider, {"model": model})


def test_ollama_needs_no_key_and_conforms():
    pytest.importorskip("langchain_ollama")
    llm = registry.create("llm", "ollama", {"model": "llama3.1"})
    assert isinstance(llm, LLM) and llm.name == "ollama:llama3.1"


def test_claude_has_no_temperature_because_current_models_reject_it(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    with pytest.raises(ValidationError):  # asking for one is an error here, not a 400 at run time
        registry.create("llm", "claude", {"model": "claude-opus-5", "temperature": 0.0})
    config = RAGConfig.from_dict(
        {"generation": {"llm": {"provider": "claude", "params": {"model": "claude-opus-5"}}}}
    )
    # the SDK's default generation temperature must not be injected into a provider that can't take it
    llm = build_llm(config, config.generation.llm, None, defaults=generation_defaults(config))
    assert llm.name == "claude:claude-opus-5"


def test_claude_output_ceiling_is_configurable(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    config = RAGConfig.from_dict(
        {
            "generation": {
                "llm": {"provider": "claude", "params": {"model": "m"}},
                "max_output_tokens": 500,
            }
        }
    )
    llm = build_llm(config, config.generation.llm, None, defaults=generation_defaults(config))
    assert llm.inner._model.max_tokens == 500


def test_openai_accepts_a_null_temperature_and_a_custom_endpoint(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    llm = registry.create(
        "llm", "openai", {"model": "m", "temperature": None, "base_url": "http://localhost:8000/v1"}
    )
    assert llm._model.temperature is None


def test_typos_in_new_provider_params_are_rejected(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    with pytest.raises(ValidationError):
        registry.create("llm", "openai", {"model": "m", "temprature": 0.1})


def test_embedders_construct_offline_and_conform(monkeypatch):
    pytest.importorskip("langchain_openai")
    pytest.importorskip("langchain_ollama")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    openai = registry.create("embedder", "openai", {"dimensions": 256})
    assert isinstance(openai, Embedder) and openai.dimensions == 256 and "256" in openai.model_id
    ollama = registry.create("embedder", "ollama", {})
    assert isinstance(ollama, Embedder) and ollama.model_id == "ollama:nomic-embed-text"
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        registry.create("embedder", "openai", {})


async def test_openai_embedder_learns_its_dimension_from_a_response(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    embedder = registry.create("embedder", "openai", {})

    class Fake:
        async def aembed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        async def aembed_query(self, text):
            return [0.1, 0.2, 0.3]

    embedder._model = Fake()
    assert embedder.dimensions == 0
    assert len(await embedder.embed_documents(["a", "b"])) == 2 and embedder.dimensions == 3


def test_every_preset_names_only_providers_that_exist():
    """A preset that references a provider nobody built would fail on first use."""
    stage_of = {
        "llm": "llm",
        "embedder": "embedder",
        "store": "store",
        "chunker": "chunker",
        "cache": "cache",
    }
    for name in available_presets():
        config = RAGConfig.preset(name)
        components = {
            "llm": [config.generation.llm, config.evaluation.judge_llm],
            "embedder": [config.embedder],
            "store": [config.store],
            "chunker": [config.chunker],
            "cache": [config.cache],
        }
        for stage, items in components.items():
            for component in filter(None, items):
                registry.resolve(stage_of[stage], component.provider)  # raises if unregistered
        if config.retrieval.reranker:
            registry.resolve("reranker", config.retrieval.reranker.provider)
        for transform in config.retrieval.query_transforms:
            registry.resolve("query_transform", transform.provider)
