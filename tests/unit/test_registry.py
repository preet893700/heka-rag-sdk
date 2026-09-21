import pytest
from pydantic import BaseModel, ValidationError

from heka.rag import MissingExtraError, UnknownProviderError
from heka.rag.registry import Registry


class Settings(BaseModel):
    model: str
    size: int = 3


def make() -> Registry:
    reg = Registry()
    reg._plugins_loaded = True  # keep tests independent of whatever is installed
    return reg


def test_register_and_create_validates_settings():
    reg = make()

    @reg.register("embedder", "fake")
    class Fake:
        settings_model = Settings

        def __init__(self, settings: Settings) -> None:
            self.settings = settings

    built = reg.create("embedder", "fake", {"model": "m"})
    assert built.settings.size == 3
    with pytest.raises(ValidationError):
        reg.create("embedder", "fake", {"size": 1})  # missing required 'model'


def test_create_without_settings_model_passes_kwargs():
    reg = make()

    @reg.register("cache", "plain")
    class Plain:
        def __init__(self, ttl: int = 1) -> None:
            self.ttl = ttl

    assert reg.create("cache", "plain", {"ttl": 9}).ttl == 9


def test_unknown_provider_lists_available():
    reg = make()
    reg.register("llm", "a")(type("A", (), {}))
    with pytest.raises(UnknownProviderError, match=r"Available: a"):
        reg.resolve("llm", "b")


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="Unknown stage"):
        make().names("nonsense")


def test_lazy_registration_defers_import():
    reg = make()
    reg.register_lazy("cache", "lazy", "collections:OrderedDict")
    assert reg.resolve("cache", "lazy").__name__ == "OrderedDict"


def test_missing_optional_dependency_gives_install_hint():
    reg = make()
    reg.register_lazy("loader", "pdf", "definitely_not_installed_pkg:Loader", extra="pdf")
    with pytest.raises(MissingExtraError, match=r"pip install 'heka-rag-sdk\[pdf\]'"):
        reg.resolve("loader", "pdf")


def test_lazy_target_must_have_attr_separator():
    with pytest.raises(ValueError, match="module:ClassName"):
        make().register_lazy("cache", "bad", "no_separator")


def test_settings_schema_for_admin_ui():
    reg = make()
    reg.register("embedder", "fake")(type("F", (), {"settings_model": Settings}))
    schema = reg.settings_schema("embedder", "fake")
    assert schema is not None and "model" in schema["properties"]
    reg.register("cache", "none")(type("N", (), {}))
    assert reg.settings_schema("cache", "none") is None
