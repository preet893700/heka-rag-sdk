import json

import pytest

from heka.rag import HekaRagError
from heka.rag.cli import _load_variants, main
from heka.rag.eval import EvalCase, EvalDataset, SourceRef, run_ablation
from heka.rag.eval.ablation import variant_config


def dataset():
    return EvalDataset(
        name="bench",
        cases=[
            EvalCase(
                id="casual",
                question="How many days of casual leave?",
                gold_answer="12 days",
                gold_sources=[SourceRef(source="leave.md", quote="12 days of casual leave")],
                tags=["leave"],
                split="dev",
            ),
            EvalCase(
                id="claims",
                question="When must expense claims be submitted?",
                gold_answer="30 days",
                gold_sources=[SourceRef(source="expenses.md", quote="within 30 days")],
                tags=["expenses"],
                split="test",
            ),
            EvalCase(id="zebra", question="zebra?", answerable=False),
        ],
    )


def test_variant_config_merges_overrides_and_gets_its_own_index(make_config):
    base = make_config()
    variant = variant_config(base, "Hybrid v2", {"retrieval": {"mode": "hybrid", "top_k": 12}})
    assert variant.retrieval.mode == "hybrid" and variant.retrieval.top_k == 12
    assert variant.retrieval.final_k == base.retrieval.final_k  # untouched values are inherited
    assert variant.knowledge.persist_dir.endswith("ablation-hybrid-v2")
    assert variant.knowledge.persist_dir != base.knowledge.persist_dir
    assert base.retrieval.mode == "dense"  # the base config is never mutated


def test_variants_share_one_response_cache(make_config):
    base = make_config(cache={"provider": "disk"})
    one = variant_config(base, "a", {})
    two = variant_config(base, "b", {"retrieval": {"mode": "sparse"}})
    assert one.cache.params["path"] == two.cache.params["path"]
    assert one.cache.params["path"].endswith("cache.sqlite")
    assert "ablation-" not in one.cache.params["path"]


async def test_run_ablation_compares_variants_on_the_same_questions(make_config):
    variants = {
        "dense": {"retrieval": {"mode": "dense"}},
        "sparse": {"retrieval": {"mode": "sparse"}},
        "hybrid": {"retrieval": {"mode": "hybrid"}},
    }
    report = await run_ablation(make_config(), variants, dataset(), retrieval_only=True)
    assert [v.name for v in report.variants] == ["dense", "sparse", "hybrid"]
    for variant in report.variants:
        assert variant.report is not None and variant.error is None and variant.chunks > 0
        assert variant.report.summary["retrieval_hit"] == 1.0  # tiny corpus: every mode finds it
        assert variant.llm_calls == 0
    text = report.to_text()
    assert "retrieval_hit" in text and "sparse" in text and "hybrid" in text
    assert "by question type" in text and "leave" in text and "expenses" in text
    tags = report.tag_scores("retrieval_hit")
    assert set(tags) == {"leave", "expenses"} and set(tags["leave"]) == {
        "dense",
        "sparse",
        "hybrid",
    }
    assert report.to_markdown().startswith("# Ablation: bench")


def test_deltas_are_relative_to_the_first_variant():
    from heka.rag.eval import EvalReport, VariantResult
    from heka.rag.eval.ablation import AblationReport

    def variant(name, hit):
        report = EvalReport(
            name="d", created_at="t", agent="a", results=[], summary={"retrieval_hit": hit}
        )
        return VariantResult(name=name, overrides={}, report=report)

    report = AblationReport(
        dataset="d",
        created_at="t",
        retrieval_only=True,
        variants=[variant("base", 0.80), variant("better", 0.95), variant("worse", 0.50)],
    )
    line = next(x for x in report.to_text().splitlines() if x.startswith("retrieval_hit"))
    assert "0.80" in line and "0.95 (+0.15)" in line and "0.50 (-0.30)" in line
    assert "(+0.00)" not in line and line.count("(") == 2  # the baseline column has no delta


async def test_a_broken_variant_is_skipped_and_the_rest_still_run(make_config):
    variants = {
        "ok": {},
        "bad": {"retrieval": {"reranker": {"provider": "no_such_reranker"}}},
        "also-ok": {"retrieval": {"mode": "sparse"}},
    }
    report = await run_ablation(make_config(), variants, dataset(), retrieval_only=True)
    bad = next(v for v in report.variants if v.name == "bad")
    assert bad.report is None and "no_such_reranker" in bad.error
    assert sum(1 for v in report.variants if v.report is not None) == 2
    assert "SKIPPED bad" in report.to_text()


async def test_full_pipeline_ablation_uses_agent_and_judge(make_config):
    config = make_config(evaluation={"judge_llm": {"provider": "scripted_judge"}})
    report = await run_ablation(
        config, {"a": {}, "b": {"retrieval": {"mode": "hybrid"}}}, dataset(), retrieval_only=False
    )
    for variant in report.variants:
        summary = variant.report.summary
        assert "correctness" in summary and "abstain_correct" in summary
        assert variant.llm_calls > 0


def test_variants_file_parsing(tmp_path):
    path = tmp_path / "v.yaml"
    path.write_text("a:\nb: {retrieval: {mode: sparse}}\n", encoding="utf-8")
    assert _load_variants(str(path)) == {"a": {}, "b": {"retrieval": {"mode": "sparse"}}}
    path.write_text("variants:\n  x: {}\n", encoding="utf-8")
    assert _load_variants(str(path)) == {"x": {}}
    for bad in ("[1, 2]", "a: 5", "{}"):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(HekaRagError):
            _load_variants(str(path))
    with pytest.raises(HekaRagError, match="Cannot read"):
        _load_variants(str(tmp_path / "missing.yaml"))


def test_cli_ablate_end_to_end(tmp_path, capsys):
    from conftest import CORPUS

    (tmp_path / "docs").mkdir()
    for name, text in CORPUS.items():
        (tmp_path / "docs" / name).write_text(text, encoding="utf-8")
    (tmp_path / "agent.yaml").write_text(
        "name: Ablate Agent\nembedder: {provider: hashing}\n"
        "knowledge: {sources: [{location: ./docs}], persist_dir: ./idx}\n"
        "generation: {llm: {provider: scripted}}\n",
        encoding="utf-8",
    )
    (tmp_path / "variants.yaml").write_text(
        "dense: {retrieval: {mode: dense}}\nhybrid: {retrieval: {mode: hybrid}}\n", encoding="utf-8"
    )
    questions = [
        {
            "id": "casual",
            "question": "How many days of casual leave?",
            "gold_answer": "12",
            "gold_sources": [{"source": "leave.md", "quote": "12 days of casual leave"}],
            "tags": ["leave"],
        }
    ]
    (tmp_path / "q.jsonl").write_text("\n".join(json.dumps(q) for q in questions), encoding="utf-8")
    out = tmp_path / "out"
    code = main(
        [
            "eval",
            "ablate",
            "-c",
            str(tmp_path / "agent.yaml"),
            "-d",
            str(tmp_path / "q.jsonl"),
            "--variants",
            str(tmp_path / "variants.yaml"),
            "--retrieval-only",
            "--out",
            str(out),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0 and "retrieval_hit" in captured.out and "hybrid" in captured.out
    assert (out / "ablation.json").exists() and (out / "ablation.md").exists()
    assert (tmp_path / "idx" / "ablation-dense").is_dir()  # each variant kept its own index


def test_cli_ablate_fails_when_every_variant_fails(tmp_path, capsys):
    from conftest import CORPUS

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "leave.md").write_text(CORPUS["leave.md"], encoding="utf-8")
    (tmp_path / "agent.yaml").write_text(
        "name: X\nembedder: {provider: hashing}\n"
        "knowledge: {sources: [{location: ./docs}], persist_dir: ./idx}\n"
        "generation: {llm: {provider: scripted}}\n",
        encoding="utf-8",
    )
    (tmp_path / "variants.yaml").write_text(
        "bad: {retrieval: {reranker: {provider: nope}}}\n", encoding="utf-8"
    )
    (tmp_path / "q.jsonl").write_text(
        json.dumps(
            {
                "id": "a",
                "question": "q",
                "gold_answer": "a",
                "gold_sources": [{"source": "leave.md"}],
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "eval",
            "ablate",
            "-c",
            str(tmp_path / "agent.yaml"),
            "-d",
            str(tmp_path / "q.jsonl"),
            "--variants",
            str(tmp_path / "variants.yaml"),
            "--retrieval-only",
        ]
    )
    assert code == 1 and "SKIPPED bad" in capsys.readouterr().out
