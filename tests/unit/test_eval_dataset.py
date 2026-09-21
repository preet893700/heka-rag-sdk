import json

import pytest
from pydantic import ValidationError

from heka.rag import ConfigError
from heka.rag.eval import EvalCase, EvalDataset, load_dataset

CASES = [
    {
        "id": "leave-01",
        "question": "How many casual leave days do I get?",
        "gold_answer": "12 days per year.",
        "gold_sources": [{"source": "leave.pdf", "section": "3.2"}],
        "tags": ["leave"],
        "split": "test",
    },
    {"id": "off-01", "question": "What is the CEO's salary?", "answerable": False},
]


def test_load_jsonl(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in CASES), encoding="utf-8")
    dataset = load_dataset(path)
    assert [c.id for c in dataset.cases] == ["leave-01", "off-01"]
    assert dataset.name == "q"


def test_load_json_list_and_mapping_and_yaml(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(CASES), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"name": "hr", "cases": CASES}), encoding="utf-8")
    (tmp_path / "c.yaml").write_text(
        "- id: y1\n  question: Q?\n  gold_answer: A\n  gold_sources: [{source: s.pdf}]\n",
        encoding="utf-8",
    )
    assert len(load_dataset(tmp_path / "a.json").cases) == 2
    assert load_dataset(tmp_path / "b.json").name == "hr"
    assert load_dataset(tmp_path / "c.yaml").cases[0].id == "y1"


def test_unsupported_or_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_dataset(tmp_path / "missing.jsonl")
    (tmp_path / "x.csv").write_text("a", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unsupported"):
        load_dataset(tmp_path / "x.csv")


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        EvalDataset(cases=[EvalCase(id="a", question="q"), EvalCase(id="a", question="q2")])


def test_unanswerable_case_cannot_have_sources():
    with pytest.raises(ValidationError, match="unanswerable"):
        EvalCase(id="a", question="q", answerable=False, gold_sources=[{"source": "s"}])


def test_subset_by_split_and_tag():
    dataset = EvalDataset(cases=CASES)
    assert [c.id for c in dataset.subset(split="test").cases] == ["leave-01"]
    assert [c.id for c in dataset.subset(tags=["leave"]).cases] == ["leave-01"]
    assert len(dataset.subset().cases) == 2


def test_warnings_flag_weak_datasets():
    weak = EvalDataset(cases=[EvalCase(id="a", question="q")])
    text = " ".join(weak.warnings())
    assert "no gold_answer" in text
    assert "no gold_sources" in text
    assert "abstention" in text
    assert "held-out" in text
    assert EvalDataset(cases=CASES).warnings() == []
