import json

import pytest

from conftest import CORPUS
from kbsdk.cli import main


@pytest.fixture
def project(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    config = tmp_path / "agent.yaml"
    config.write_text(
        "name: CLI Agent\n"
        "embedder: {provider: hashing}\n"
        "knowledge:\n  sources: [{location: ./docs}]\n  persist_dir: ./idx\n"
        "generation: {llm: {provider: scripted}}\n"
        "evaluation: {judge_llm: {provider: scripted_judge}}\n",
        encoding="utf-8",
    )
    questions = tmp_path / "q.jsonl"
    rows = [
        {
            "id": "casual",
            "question": "How many days of casual leave?",
            "gold_answer": "12 days",
            "gold_sources": [{"source": "leave.md", "section": "Casual leave"}],
            "split": "dev",
        },
        {
            "id": "zebra",
            "question": "zebra astronomy question",
            "answerable": False,
            "split": "test",
        },
    ]
    questions.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path, str(config), str(questions)


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_presets(capsys):
    code, out, _ = run(capsys, "presets")
    assert code == 0 and "free-tier-dev" in out.split()


def test_ingest_then_ingest_again(project, capsys):
    root, config, _ = project
    code, out, _ = run(capsys, "ingest", "-c", config)
    assert code == 0 and "2 new" in out
    assert (
        root / "idx" / "cli-agent" / "manifest.json"
    ).exists()  # paths resolve next to the config
    _, out, _ = run(capsys, "ingest", "-c", config)
    assert "2 unchanged" in out


def test_inspect_shows_retrieved_passages(project, capsys):
    _, config, _ = project
    code, out, _ = run(capsys, "inspect", "-c", config, "casual leave days", "-k", "2")
    assert code == 0 and "[1] score" in out and "leave.md" in out and "Casual leave" in out


def test_ask_prints_answer_and_citations(project, capsys):
    _, config, _ = project
    code, out, _ = run(capsys, "ask", "-c", config, "How many days of casual leave?")
    assert code == 0 and "[1] leave.md" in out and "Leave Policy > Casual leave" in out
    code, out, _ = run(capsys, "ask", "-c", config, "zebra astronomy question")
    assert code == 0 and "[abstained: no_relevant_context]" in out


def test_ask_json(project, capsys):
    _, config, _ = project
    _, out, _ = run(capsys, "ask", "-c", config, "How many days of casual leave?", "--json")
    payload = json.loads(out)
    assert payload["abstained"] is False and payload["citations"][0]["verified"] is True


def test_eval_check(project, capsys):
    _, _, questions = project
    code, out, _ = run(capsys, "eval", "check", questions)
    assert code == 0 and "2 cases (1 answerable, 1 unanswerable)" in out


def test_eval_run_saves_a_report_and_compares_to_baseline(project, capsys, tmp_path):
    _, config, questions = project
    out_dir = tmp_path / "run1"
    code, out, _ = run(capsys, "eval", "run", "-c", config, "-d", questions, "--out", str(out_dir))
    assert code == 0 and "correctness" in out and "saved:" in out
    assert (out_dir / "report.json").exists() and (out_dir / "report.md").exists()
    code, out, _ = run(
        capsys,
        "eval",
        "run",
        "-c",
        config,
        "-d",
        questions,
        "--out",
        str(tmp_path / "run2"),
        "--baseline",
        str(out_dir / "report.json"),
        "--no-judge",
    )
    assert code == 0 and "vs baseline" in out


def test_eval_run_retrieval_only_needs_no_llm(project, capsys, tmp_path, monkeypatch):
    root, _, questions = project
    # a config whose LLM provider needs a key that is not set: retrieval-only must not touch it
    (root / "nokey.yaml").write_text(
        "name: CLI Agent\nembedder: {provider: hashing}\n"
        "knowledge: {sources: [{location: ./docs}], persist_dir: ./idx}\n"
        "generation: {llm: {provider: gemini, params: {model: m}}}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    code, out, _ = run(
        capsys,
        "eval",
        "run",
        "-c",
        str(root / "nokey.yaml"),
        "-d",
        questions,
        "--retrieval-only",
        "--out",
        str(tmp_path / "r"),
    )
    assert code == 0 and "retrieval_hit" in out and "correctness" not in out
    assert (tmp_path / "r" / "report.json").exists()


def test_eval_run_filters_and_limit(project, capsys, tmp_path):
    _, config, questions = project
    _, out, _ = run(
        capsys,
        "eval",
        "run",
        "-c",
        config,
        "-d",
        questions,
        "--split",
        "test",
        "--no-judge",
        "--out",
        str(tmp_path / "r"),
    )
    assert "cases: 1 (0 answerable, 1 unanswerable" in out
    code, _, err = run(
        capsys, "eval", "run", "-c", config, "-d", questions, "--tag", "nothing", "--no-judge"
    )
    assert code == 2 and "No cases" in err


def test_eval_run_fails_the_build_when_thresholds_are_missed(project, capsys, tmp_path):
    root, config, questions = project
    text = (
        root / "agent.yaml"
    ).read_text() + "  thresholds: {answerable_abstain_rate: 0.0, unanswerable_abstain_rate: 2.0}\n"
    (root / "agent.yaml").write_text(
        text.replace(
            "evaluation: {judge_llm: {provider: scripted_judge}}\n",
            "evaluation:\n  judge_llm: {provider: scripted_judge}\n",
        )
    )
    code, out, _ = run(
        capsys,
        "eval",
        "run",
        "-c",
        config,
        "-d",
        questions,
        "--out",
        str(tmp_path / "r"),
        "--no-judge",
    )
    assert code == 1 and "thresholds: FAILED" in out


def test_resume_requires_out(project, capsys):
    _, config, questions = project
    code, _, err = run(
        capsys, "eval", "run", "-c", config, "-d", questions, "--resume", "--no-judge"
    )
    assert code == 2 and "--out" in err


def test_errors_are_friendly(project, capsys, tmp_path):
    code, _, err = run(capsys, "ask", "-c", str(tmp_path / "missing.yaml"), "hi")
    assert code == 2 and err.startswith("error: Cannot read config file")
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "preset: free-tier-dev\ngeneration: {citations: {mode: native}}\n", encoding="utf-8"
    )
    code, _, err = run(capsys, "ingest", "-c", str(bad))
    assert code == 2 and "not available yet" in err


def test_empty_index_is_reported(project, capsys, tmp_path):
    root, config, _ = project
    for path in (root / "docs").iterdir():
        path.unlink()
    code, _, err = run(capsys, "ask", "-c", config, "anything")
    assert code == 2 and "index is empty" in err
