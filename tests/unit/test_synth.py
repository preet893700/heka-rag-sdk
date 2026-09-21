import json
import re
from pathlib import Path

import pytest

from conftest import ScriptedLLM
from kbsdk import KnowledgeBase, RequestContext, registry
from kbsdk.cli import main
from kbsdk.eval import SynthOptions, generate_dataset, load_dataset
from kbsdk.eval.synth import copies_wording, numbers_supported, too_similar

FIRST_SENTENCE = re.compile(r"(?<=[.!?])\s")

# What a well-behaved model writes for each passage in the shared test corpus.
DRAFTS = {
    "casual": ("How much casual time off do I get?", "12 days"),
    "sick": ("What time off is there when I am ill?", "10 days"),
    "30 days": ("By when must I file my claims?", "30 days"),
    "1000": ("Who signs off small claims?", "Managers, up to 1000 dollars"),
}
FOLLOWUPS = {
    "casual": ("And does it carry over?", "No, unused casual leave does not carry over."),
    "sick": ("What proof do I need?", "A medical certificate after 2 consecutive days."),
    "30 days": ("What happens to late ones?", "Older claims are rejected."),
}


def passage_of(messages):
    return messages[-1].content.split("Passage:\n", 1)[1].strip()


def draft_for(passage, table):
    return next((v for k, v in table.items() if k in passage), None)


def make_reply(mode="good", asked_unanswerable=None):
    """A scripted 'model' for all four synthetic-data prompts; `mode` injects one kind of mistake."""

    def reply(messages, system):
        if "Write ONE realistic question" in system:
            passage = passage_of(messages)
            found = draft_for(passage, DRAFTS)
            if found is None:
                return json.dumps({"skip": True})
            question, answer = found
            quote = FIRST_SENTENCE.split(passage)[0]
            if mode == "bad_quote":
                quote = "a sentence the document never contains"
            elif mode == "copied":
                question = FIRST_SENTENCE.split(passage)[0].rstrip(".") + ", right?"
            elif mode == "bad_number":
                answer = "97 days"
            elif mode == "refers":
                question = "What does the passage say about time off?"
            elif mode == "same":
                question = "How much time off do I get?"
            elif mode == "junk":
                return "not json at all"
            return json.dumps({"question": question, "answer": answer, "quote": quote})
        if "two-turn exchange" in system:
            passage = passage_of(messages)
            found = draft_for(passage, FOLLOWUPS)
            first = draft_for(passage, DRAFTS)
            if found is None or first is None:
                return json.dumps({"skip": True})
            quote = (
                FIRST_SENTENCE.split(passage)[-1]
                if mode != "bad_quote"
                else "never written anywhere"
            )
            return json.dumps(
                {
                    "question": first[0],
                    "answer": first[1],
                    "followup": found[0],
                    "followup_answer": found[1],
                    "quote": quote,
                }
            )
        if "outline of the company documents" in system:
            return json.dumps(
                {
                    "questions": [
                        "What is the parental leave policy?",
                        "Can I claim gym membership costs?",
                        "How many days of sick leave do I get?",  # in fact answered by the documents
                    ]
                }
            )
        if "whether any of the numbered passages" in system:
            question = messages[-1].content.rsplit("Question:", 1)[-1].lower()
            answered = "sick" in question
            if asked_unanswerable is not None:
                asked_unanswerable.append(question.strip())
            return json.dumps({"answered": answered})
        raise AssertionError(f"unexpected prompt: {system[:60]}")

    return reply


OPTIONS = dict(min_chunk_chars=20, concurrency=2)


async def generate(kb, mode="good", **overrides):
    llm = ScriptedLLM(make_reply(mode), name="fake:writer")
    options = SynthOptions(
        **{**OPTIONS, "answerable": 2, "followups": 1, "unanswerable": 2, **overrides}
    )
    return await generate_dataset(kb, llm, options, name="gen")


# -- the deterministic checks --------------------------------------------------------------------


def test_copies_wording_needs_a_long_run_of_shared_words():
    passage = "Employees receive 12 days of casual leave per calendar year."
    assert copies_wording("Do employees receive 12 days of casual leave?", passage)
    assert not copies_wording("How many days off can staff take?", passage)
    assert not copies_wording("employees receive 12 days", passage)  # too short a run to count


def test_numbers_in_a_reference_answer_must_come_from_the_passage():
    passage = "Managers approve claims up to 1,000 dollars within 30 days."
    assert numbers_supported("Up to 1000 dollars", passage)  # thousands separator ignored
    assert numbers_supported("30 days", passage)
    assert numbers_supported("Managers approve them", passage)  # no numbers, nothing to check
    assert not numbers_supported("Up to 5000 dollars", passage)


def test_near_duplicate_questions_are_detected():
    assert too_similar(
        "How many casual leave days do I get?", ["How many casual leave days do I get"]
    )
    assert not too_similar("Who approves expenses?", ["How many casual leave days do I get?"])
    assert too_similar("???", [])  # nothing left to compare: unusable


# -- generation ----------------------------------------------------------------------------------


async def test_generates_answerable_followup_and_unanswerable_cases(kb):
    result = await generate(kb)
    cases = {c.id: c for c in result.dataset.cases}
    assert result.kept == {"answerable": 2, "followup": 1, "unanswerable": 2}
    assert set(cases) == {"syn-a-001", "syn-a-002", "syn-f-001", "syn-u-001", "syn-u-002"}

    answerable = cases["syn-a-001"]
    ref = answerable.gold_sources[0]
    assert answerable.answerable and answerable.gold_answer and "synthetic" in answerable.tags
    assert ref.source in {"leave.md", "expenses.md"} and ref.quote and ref.section

    followup = cases["syn-f-001"]
    assert [m.role for m in followup.history] == ["user", "assistant"]
    assert "followup" in followup.tags and len(followup.question.split()) <= 14

    for case in (cases["syn-u-001"], cases["syn-u-002"]):
        assert not case.answerable and not case.gold_sources and "unanswerable" in case.tags


async def test_the_generated_gold_quotes_really_are_in_the_index(kb):
    from kbsdk.text import quote_in_text

    result = await generate(kb, answerable=4, followups=0, unanswerable=0)
    chunks = await kb.store.all_chunks()
    assert result.kept["answerable"] == 4
    for case in result.dataset.cases:
        ref = case.gold_sources[0]
        assert any(
            quote_in_text(ref.quote, c.text) and c.metadata["source"] == ref.source for c in chunks
        )


async def test_unanswerable_questions_the_documents_actually_answer_are_dropped(kb):
    asked: list[str] = []
    llm = ScriptedLLM(make_reply("good", asked), name="fake:writer")
    options = SynthOptions(**OPTIONS, answerable=0, followups=0, unanswerable=3)
    result = await generate_dataset(kb, llm, options)
    assert [c.question for c in result.dataset.cases] == [
        "What is the parental leave policy?",
        "Can I claim gym membership costs?",
    ]
    assert result.dropped == {"unanswerable: actually_answerable": 1}
    assert any("sick leave" in q for q in asked)  # every proposal was checked against retrieval


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("bad_quote", "quote_not_in_passage"),
        ("copied", "copies_passage_wording"),
        ("bad_number", "answer_number_not_in_passage"),
        ("refers", "refers_to_passage"),
        ("junk", "unusable_reply"),
    ],
)
async def test_bad_drafts_are_dropped_and_reported_by_reason(kb, mode, reason):
    result = await generate(kb, mode, followups=0, unanswerable=0)
    answerable_kept = [c for c in result.dataset.cases if c.answerable]
    assert answerable_kept == [] and result.kept["answerable"] == 0
    assert result.dropped[f"answerable: {reason}"] >= 1
    assert any("only 0 of 2 answerable" in w for w in result.warnings)


async def test_a_followup_whose_quote_is_not_in_the_passage_is_dropped(kb):
    result = await generate(kb, "bad_quote", answerable=0, unanswerable=0)
    assert result.kept["followup"] == 0
    assert result.dropped["followup: quote_not_in_passage"] >= 1


async def test_duplicate_questions_are_kept_once(kb):
    result = await generate(kb, "same", answerable=4, followups=0, unanswerable=0)
    assert result.kept["answerable"] == 1
    assert result.dropped["answerable: duplicate_question"] == 3


async def test_unsuitable_passages_are_skipped_not_fatal(kb):
    result = await generate(kb, answerable=4, followups=0, unanswerable=0, min_chunk_chars=20)
    assert result.kept["answerable"] == 4  # every passage in the corpus is usable
    tiny = await generate(kb, answerable=2, followups=0, unanswerable=0, min_chunk_chars=10_000)
    assert tiny.dataset.cases == [] and "no passages" in tiny.warnings[0]


async def test_same_seed_same_dataset_and_splits_are_stable(kb):
    first = await generate(kb, test_fraction=0.5, seed=3)
    second = await generate(kb, test_fraction=0.5, seed=3)
    assert first.dataset.model_dump() == second.dataset.model_dump()
    assert {c.split for c in first.dataset.cases} <= {"dev", "test"}
    none = await generate(kb, test_fraction=0.0)
    assert {c.split for c in none.dataset.cases} == {"dev"}
    every = await generate(kb, test_fraction=1.0)
    assert {c.split for c in every.dataset.cases} == {"test"}


async def test_followups_get_first_pick_of_a_small_pool_and_the_shortfall_is_explained(kb):
    result = await generate(kb, answerable=4, followups=1, unanswerable=0)
    assert (
        result.kept["followup"] == 1
    )  # would be 0 if answerable questions used every passage first
    assert result.kept["answerable"] == 3
    shortfall = [w for w in result.warnings if "answerable" in w]
    assert shortfall and "ran out of passages" in shortfall[0]


async def test_coverage_is_reported(kb):
    result = await generate(kb, answerable=1, followups=0, unanswerable=0)
    assert result.sources_total == 2 and result.sources_covered == 1
    assert any("cover 1 of 2 documents" in w for w in result.warnings)


async def test_only_passages_the_caller_may_see_are_used(make_config, docs_dir):
    knowledge = make_config().to_dict()["knowledge"]
    knowledge["sources"] = [
        {"location": str(docs_dir / "leave.md"), "metadata": {"roles": ["hr"]}},
        {"location": str(docs_dir / "expenses.md"), "metadata": {"roles": ["*"]}},
    ]
    kb = KnowledgeBase(make_config(knowledge=knowledge, access={"roles_field": "roles"}))
    await kb.aingest()
    employee = RequestContext(roles=["employee"])
    result = await generate(kb, context=employee, answerable=4, followups=0, unanswerable=0)
    sources = {c.gold_sources[0].source for c in result.dataset.cases}
    assert sources == {"expenses.md"}  # HR-only leave.md was never shown to the model
    assert all(c.context == employee for c in result.dataset.cases)
    hr = await generate(
        kb, context=RequestContext(roles=["hr"]), answerable=4, followups=0, unanswerable=0
    )
    assert {c.gold_sources[0].source for c in hr.dataset.cases} == {"leave.md", "expenses.md"}


# -- CLI -----------------------------------------------------------------------------------------


class WriterLLM(ScriptedLLM):
    def __init__(self):
        super().__init__(make_reply(), name="fake:writer")


registry.register("llm", "scripted_writer")(WriterLLM)


@pytest.fixture
def project(tmp_path):
    from conftest import CORPUS

    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    config = tmp_path / "agent.yaml"
    config.write_text(
        "name: Gen Agent\n"
        "embedder: {provider: hashing}\n"
        "knowledge:\n  sources: [{location: ./docs}]\n  persist_dir: ./idx\n"
        "generation: {llm: {provider: scripted}}\n"
        "evaluation: {generator_llm: {provider: scripted_writer}}\n",
        encoding="utf-8",
    )
    return tmp_path, str(config)


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


ARGS = ["--count", "2", "--followups", "1", "--unanswerable", "3", "--min-chars", "20"]


def test_cli_writes_a_dataset_that_loads_and_checks(project, capsys):
    root, config = project
    out = root / "gen" / "questions.jsonl"
    code, stdout, _ = run(capsys, "eval", "gen", "-c", config, "-o", str(out), *ARGS)
    assert code == 0, stdout
    assert "wrote 5 questions" in stdout and "dropped by the checks" in stdout
    assert "actually_answerable" in stdout and "synthetic questions echo" in stdout
    assert "llm calls:" in stdout and "fake:writer" in stdout
    assert "review these" in stdout and "syn-u-001: What is the parental leave policy?" in stdout
    dataset = load_dataset(out)
    assert len(dataset.cases) == 5 and dataset.name == "questions"
    code, stdout, _ = run(capsys, "eval", "check", str(out))
    assert code == 0 and "3 answerable, 2 unanswerable" in stdout
    # the generated file works with the runner: retrieval-only needs no LLM
    code, stdout, _ = run(
        capsys,
        "eval",
        "run",
        "-c",
        config,
        "-d",
        str(out),
        "--retrieval-only",
        "--out",
        str(root / "r"),
    )
    assert code == 0 and "retrieval_hit" in stdout


def test_cli_refuses_to_overwrite_without_force(project, capsys):
    root, config = project
    out = root / "q.jsonl"
    out.write_text("keep me", encoding="utf-8")
    code, _, err = run(capsys, "eval", "gen", "-c", config, "-o", str(out), *ARGS)
    assert code == 2 and "already exists" in err and out.read_text(encoding="utf-8") == "keep me"
    code, _, _ = run(capsys, "eval", "gen", "-c", config, "-o", str(out), "--force", *ARGS)
    assert code == 0 and out.read_text(encoding="utf-8") != "keep me"


def test_cli_warns_when_the_tested_model_writes_its_own_questions(project, capsys):
    root, config = project
    text = Path(config).read_text(encoding="utf-8")
    text = text.replace(
        "generation: {llm: {provider: scripted}}", "generation: {llm: {provider: scripted_writer}}"
    )
    text = text.replace("evaluation: {generator_llm: {provider: scripted_writer}}\n", "")
    Path(config).write_text(text, encoding="utf-8")
    _, stdout, _ = run(capsys, "eval", "gen", "-c", config, "-o", str(root / "q.jsonl"), *ARGS)
    assert "is the model being tested" in stdout


def test_cli_needs_an_identity_when_access_control_is_on(project, capsys):
    root, config = project
    text = (
        Path(config)
        .read_text(encoding="utf-8")
        .replace(
            "knowledge:\n  sources: [{location: ./docs}]",
            "knowledge:\n  sources: [{location: ./docs, metadata: {roles: ['*']}}]",
        )
    )
    Path(config).write_text(text + "access: {roles_field: roles}\n", encoding="utf-8")
    out = root / "q.jsonl"
    code, _, err = run(capsys, "eval", "gen", "-c", config, "-o", str(out), *ARGS)
    assert code == 2 and "--context" in err and not out.exists()
    code, stdout, _ = run(
        capsys,
        "eval",
        "gen",
        "-c",
        config,
        "-o",
        str(out),
        "--context",
        '{"roles": ["employee"]}',
        *ARGS,
    )
    assert code == 0, stdout
    assert all(c.context and c.context.roles == ["employee"] for c in load_dataset(out).cases)
    code, _, err = run(
        capsys, "eval", "gen", "-c", config, "-o", str(out), "--force", "--context", "{nope", *ARGS
    )
    assert code == 2 and "--context must be a JSON object" in err
