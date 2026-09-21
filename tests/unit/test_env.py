import json
import os

import pytest

from heka.rag import ConfigError, RAGConfig, load_env_file
from heka.rag.cli import main
from heka.rag.env import parse_env_text


def test_parsing_syntax_variants():
    text = "\n".join(
        [
            "# a comment",
            "",
            "PLAIN=value",
            "  SPACED  =  padded value  ",
            "export EXPORTED=yes",
            'DOUBLE="has spaces # not a comment"',
            "SINGLE='literal $HOME \\n'",
            r'ESCAPED="line1\nline2 \"quoted\" back\\slash"',
            "INLINE=value # trailing comment",
            "HASHED=abc#not-a-comment",
            "EMPTY=",
            'AFTER_QUOTE="kept" # comment after',
            "DUP=first",
            "DUP=second",
            "dotted.name-1=ok",
        ]
    )
    values = parse_env_text(text)
    assert values["PLAIN"] == "value"
    assert values["SPACED"] == "padded value"
    assert values["EXPORTED"] == "yes"
    assert values["DOUBLE"] == "has spaces # not a comment"
    assert values["SINGLE"] == "literal $HOME \\n"  # single quotes are fully literal
    assert values["ESCAPED"] == 'line1\nline2 "quoted" back\\slash'
    assert values["INLINE"] == "value"
    assert values["HASHED"] == "abc#not-a-comment"
    assert values["EMPTY"] == ""
    assert values["AFTER_QUOTE"] == "kept"
    assert values["DUP"] == "second"
    assert values["dotted.name-1"] == "ok"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not an assignment", r":2: expected KEY=value"),
        ('A="open-secret', r":2: unterminated double"),
        ("A='open-secret", r":2: unterminated single"),
    ],
)
def test_malformed_lines_report_the_line_number_but_not_the_value(text, message):
    with pytest.raises(ConfigError, match=message) as info:
        parse_env_text(f"OK=1\n{text}")
    assert "open-secret" not in str(info.value)  # never echoes the offending value


def test_load_sets_variables_and_reports_names_only(tmp_path, monkeypatch):
    monkeypatch.delenv("HEKA_RAG_T_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("HEKA_RAG_T_KEY=super-secret-value\n", encoding="utf-8")
    result = load_env_file(path)
    assert os.environ["HEKA_RAG_T_KEY"] == "super-secret-value"
    assert result.loaded == ["HEKA_RAG_T_KEY"] and result.skipped_existing == []
    assert "super-secret-value" not in result.summary() and "HEKA_RAG_T_KEY" in result.summary()
    monkeypatch.delenv("HEKA_RAG_T_KEY")


def test_existing_environment_wins_unless_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HEKA_RAG_T_KEY", "from-shell")
    path = tmp_path / ".env"
    path.write_text("HEKA_RAG_T_KEY=from-file\n", encoding="utf-8")
    result = load_env_file(path)
    assert os.environ["HEKA_RAG_T_KEY"] == "from-shell"
    assert result.skipped_existing == ["HEKA_RAG_T_KEY"] and result.loaded == []
    load_env_file(path, override=True)
    assert os.environ["HEKA_RAG_T_KEY"] == "from-file"


def test_missing_file_is_an_error_unless_optional(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_env_file(tmp_path / "nope.env")
    assert load_env_file(tmp_path / "nope.env", must_exist=False).loaded == []


def test_bom_is_tolerated(tmp_path, monkeypatch):
    monkeypatch.delenv("HEKA_RAG_T_BOM", raising=False)
    (tmp_path / ".env").write_bytes("﻿HEKA_RAG_T_BOM=1\n".encode())
    load_env_file(tmp_path / ".env")
    assert os.environ["HEKA_RAG_T_BOM"] == "1"
    monkeypatch.delenv("HEKA_RAG_T_BOM")


def test_nothing_is_read_implicitly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEKA_RAG_T_IMPLICIT", raising=False)
    (tmp_path / ".env").write_text("HEKA_RAG_T_IMPLICIT=1\n", encoding="utf-8")
    RAGConfig.from_dict({"generation": {"llm": {"provider": "scripted"}}})
    assert "HEKA_RAG_T_IMPLICIT" not in os.environ  # a .env in the working directory is ignored


def test_config_env_file_is_relative_to_the_config_and_loaded_by_the_sdk(tmp_path, monkeypatch):
    from conftest import CORPUS
    from heka.rag import KnowledgeBase

    monkeypatch.delenv("HEKA_RAG_T_CFG", raising=False)
    (tmp_path / "docs").mkdir()
    for name, text in CORPUS.items():
        (tmp_path / "docs" / name).write_text(text, encoding="utf-8")
    (tmp_path / "secrets.env").write_text("HEKA_RAG_T_CFG=loaded-by-config\n", encoding="utf-8")
    (tmp_path / "agent.yaml").write_text(
        "name: Env Agent\nenv_file: ./secrets.env\nembedder: {provider: hashing}\n"
        "knowledge: {sources: [{location: ./docs}], persist_dir: ./idx}\n"
        "generation: {llm: {provider: scripted}}\n",
        encoding="utf-8",
    )
    config = RAGConfig.from_file(tmp_path / "agent.yaml")
    assert config.env_file == str(tmp_path / "secrets.env")
    assert "HEKA_RAG_T_CFG" not in os.environ  # loading the config alone reads nothing
    KnowledgeBase(config)
    assert os.environ["HEKA_RAG_T_CFG"] == "loaded-by-config"
    monkeypatch.delenv("HEKA_RAG_T_CFG")


def test_config_pointing_at_a_missing_env_file_fails_clearly(tmp_path):
    from heka.rag import KnowledgeBase

    config = RAGConfig.from_dict(
        {"env_file": str(tmp_path / "gone.env"), "generation": {"llm": {"provider": "scripted"}}}
    )
    with pytest.raises(ConfigError, match="Env file not found"):
        KnowledgeBase(config)


def test_cli_env_file_flag_loads_the_file_and_names_only_are_reported(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("HEKA_RAG_T_CLI", raising=False)
    (tmp_path / ".env").write_text("HEKA_RAG_T_CLI=cli-secret-value\n", encoding="utf-8")
    assert main(["presets", "--env-file", str(tmp_path / ".env")]) == 0
    captured = capsys.readouterr()
    assert os.environ["HEKA_RAG_T_CLI"] == "cli-secret-value"
    assert (
        "HEKA_RAG_T_CLI" in captured.err and "cli-secret-value" not in captured.err + captured.out
    )
    monkeypatch.delenv("HEKA_RAG_T_CLI")


def test_cli_without_the_flag_reads_no_env_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEKA_RAG_T_CLI2", raising=False)
    (tmp_path / ".env").write_text("HEKA_RAG_T_CLI2=1\n", encoding="utf-8")
    assert main(["presets"]) == 0
    assert "HEKA_RAG_T_CLI2" not in os.environ


def test_cli_env_file_missing_is_friendly(tmp_path, capsys):
    code = main(["presets", "--env-file", str(tmp_path / "missing.env")])
    err = capsys.readouterr().err
    assert code == 2 and "Env file not found" in err


def test_error_output_never_contains_secret_values(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "GOOGLE_API_KEY=sk-should-never-be-printed\nBROKEN LINE\n", encoding="utf-8"
    )
    code = main(["presets", "--env-file", str(tmp_path / ".env")])
    captured = capsys.readouterr()
    assert code == 2 and "sk-should-never-be-printed" not in captured.err + captured.out
    assert json.dumps(os.environ.get("GOOGLE_API_KEY")) == "null"  # a failed load sets nothing
