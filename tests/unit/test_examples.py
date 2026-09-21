"""The shipped example configs must stay valid and runnable as the SDK evolves."""

import json
from pathlib import Path

import pytest

from heka.rag import KnowledgeBase, RAGConfig, RequestContext, registry

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
CONFIGS = sorted(p for p in EXAMPLES.glob("*/*.yaml") if "variants" not in p.name)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_example_config_loads_and_uses_registered_providers(path):
    config = RAGConfig.from_file(path)
    registry.resolve("llm", config.generation.llm.provider)
    registry.resolve("embedder", config.embedder.provider)
    registry.resolve("chunker", config.chunker.provider)
    for guard in config.guardrails:
        registry.resolve("guardrail", guard.provider)
    for tracer in config.tracing:
        registry.resolve("tracer", tracer.provider)


async def test_the_secure_hr_example_enforces_its_own_policy(tmp_path):
    """Run hr-secure.yaml's access rules against the real example documents (no LLM, no keys)."""
    config = RAGConfig.from_file(EXAMPLES / "hr_agent" / "hr-secure.yaml")
    data = config.to_dict()
    data["embedder"] = {"provider": "hashing"}
    data["knowledge"]["persist_dir"] = str(tmp_path / "idx")
    data["retrieval"] = {"mode": "hybrid", "final_k": 10, "top_k": 20}
    kb = KnowledgeBase(RAGConfig.from_dict(data))
    report = await kb.aingest()
    assert not report.failed and not any("nobody can see it" in w for w in report.warnings)

    def visible(roles):
        return {
            c.chunk.metadata["source"]
            for c in kb.retrieve(
                "harassment report conflicts of interest leave",
                k=10,
                context=RequestContext(tenant_id="acme", roles=roles),
            )
        }

    assert "code-of-conduct.md" not in visible(["employee"])  # narrowed to hr and managers
    assert "code-of-conduct.md" in visible(["hr"])
    assert "leave-policy.md" in visible(["employee"])  # public within the tenant


def test_example_datasets_are_valid_json_lines():
    for path in EXAMPLES.glob("*/questions.jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert rows and all("id" in r and "question" in r for r in rows), path
