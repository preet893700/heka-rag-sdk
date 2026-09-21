import json

import pytest
from pydantic import ValidationError

from kbsdk import ConfigError, RAGConfig, available_presets
from kbsdk.config import ComponentConfig, deep_merge

LLM = {"provider": "gemini", "params": {"model": "m", "api_key_env": "GOOGLE_API_KEY"}}


def minimal(**extra):
    return {"generation": {"llm": LLM}, **extra}


def test_presets_exist():
    assert {"free-tier-dev", "high-accuracy", "low-cost", "on-prem"} <= set(available_presets())


@pytest.mark.parametrize("name", available_presets())
def test_every_preset_is_valid(name):
    config = RAGConfig.preset(name)
    assert config.generation.llm.provider
    assert config.retrieval.final_k <= config.retrieval.top_k


def test_defaults_give_a_working_shape():
    config = RAGConfig.from_dict(minimal())
    assert config.embedder.provider == "local"
    assert config.store.provider == "local"
    assert config.chunker.provider == "structure_aware"
    assert config.retrieval.mode == "dense"
    assert config.knowledge.update_mode == "incremental"


def test_llm_is_required():
    with pytest.raises(ValidationError):
        RAGConfig.from_dict({})


def test_preset_overrides_merge_deeply():
    config = RAGConfig.preset(
        "free-tier-dev",
        name="HR",
        knowledge={"sources": [{"location": "./docs/hr"}]},
        retrieval={"final_k": 3},
    )
    assert config.name == "HR"
    assert config.knowledge.sources[0].location == "./docs/hr"
    assert config.retrieval.final_k == 3
    assert config.retrieval.top_k == 20  # untouched preset value survives
    assert config.generation.llm.provider == "gemini"  # so do nested components


def test_switching_provider_replaces_params():
    config = RAGConfig.preset(
        "free-tier-dev",
        generation={"llm": {"provider": "openai", "params": {"model": "x"}}},
    )
    assert config.generation.llm.provider == "openai"
    assert config.generation.llm.params == {"model": "x"}  # no leaked api_key_env from gemini


def test_same_provider_merges_params():
    config = RAGConfig.preset(
        "free-tier-dev",
        generation={"llm": {"provider": "gemini", "params": {"model": "other"}}},
    )
    assert config.generation.llm.params["model"] == "other"
    assert config.generation.llm.params["api_key_env"] == "GOOGLE_API_KEY"


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}}


@pytest.mark.parametrize(
    "key", ["api_key", "openai_api_key", "secret", "password", "access_token", "token"]
)
def test_secrets_are_rejected(key):
    with pytest.raises(ValidationError, match="secrets must not be stored"):
        ComponentConfig(provider="x", params={key: "abc"})


def test_nested_secrets_are_rejected():
    with pytest.raises(ValidationError, match="secrets must not be stored"):
        ComponentConfig(provider="x", params={"auth": {"api_key": "abc"}})


@pytest.mark.parametrize("key", ["api_key_env", "max_tokens", "top_k", "model", "base_url"])
def test_normal_params_are_allowed(key):
    ComponentConfig(provider="x", params={key: "ok"})


def test_final_k_cannot_exceed_top_k():
    with pytest.raises(ValidationError, match="final_k"):
        RAGConfig.from_dict(minimal(retrieval={"top_k": 3, "final_k": 5}))


def test_typos_are_caught():
    with pytest.raises(ValidationError):
        RAGConfig.from_dict(minimal(retreival={"mode": "dense"}))


def test_unknown_preset_lists_available():
    with pytest.raises(ConfigError, match="free-tier-dev"):
        RAGConfig.preset("nope")


def test_from_file_yaml_toml_json(tmp_path):
    yaml_file = tmp_path / "a.yaml"
    yaml_file.write_text(
        "preset: low-cost\nname: From YAML\nknowledge:\n  sources:\n    - location: ./docs\n",
        encoding="utf-8",
    )
    toml_file = tmp_path / "a.toml"
    toml_file.write_text(
        'name = "From TOML"\n[generation.llm]\nprovider = "gemini"\n', encoding="utf-8"
    )
    json_file = tmp_path / "a.json"
    json_file.write_text(json.dumps(minimal(name="From JSON")), encoding="utf-8")

    assert RAGConfig.from_file(yaml_file).name == "From YAML"
    assert RAGConfig.from_file(toml_file).name == "From TOML"
    assert RAGConfig.from_file(json_file).name == "From JSON"


def test_from_file_errors(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read"):
        RAGConfig.from_file(tmp_path / "missing.yaml")
    bad = tmp_path / "a.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unsupported"):
        RAGConfig.from_file(bad)


def test_round_trip():
    config = RAGConfig.preset("high-accuracy", name="X")
    assert RAGConfig.from_dict(config.to_dict()) == config


def test_json_schema_available_for_admin_ui():
    schema = RAGConfig.json_schema()
    assert "generation" in schema["properties"]
