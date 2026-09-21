import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rename_package.py"
spec = importlib.util.spec_from_file_location("rename_package", SCRIPT)
rename = importlib.util.module_from_spec(spec)
sys.modules["rename_package"] = rename  # dataclasses looks the module up by name
spec.loader.exec_module(rename)


@pytest.fixture
def repo(tmp_path):
    def write(rel, text):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    write(
        "pyproject.toml",
        '[project]\nname = "kbsdk"  # placeholder name - rename before the first release\n'
        '[project.scripts]\nkbsdk = "kbsdk.cli:main"\n',
    )
    write(
        "src/kbsdk/__init__.py",
        '"""kbsdk - an SDK (placeholder name)."""\nfrom kbsdk.errors import KbsdkError\n',
    )
    write(
        "src/kbsdk/errors.py",
        'class KbsdkError(Exception): ...\nGROUP = "kbsdk.plugins"\nDIR = ".kbsdk"\n',
    )
    write(
        "src/kbsdk/sub/kbsdk_meter.py",
        'import os\nKEY = os.environ["KBSDK_API_KEYS"]\nfrom kbsdk import KbsdkError\n',
    )
    write("tests/test_x.py", "import kbsdk\nfrom kbsdk.errors import KbsdkError\n")
    write(
        "README.md",
        "# kbsdk\n\n> `kbsdk` is a placeholder name. Run `pip install 'kbsdk[server]'`.\n",
    )
    write("CHANGELOG.md", "# Changelog\n\n## [0.1.0] - 2026-09-21\n\nAdded kbsdk things.\n")
    write(".kbsdk/index/manifest.json", '{"note": "kbsdk state"}')
    write(".venv/lib/site.py", "kbsdk = 1")
    write(".git/config", "[remote] url = kbsdk")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00kbsdk\x80")
    return tmp_path


def read(root, rel):
    return (root / rel).read_text(encoding="utf-8")


def test_a_dry_run_changes_nothing_and_says_what_it_would_do(repo, capsys):
    before = {p: p.read_bytes() for p in repo.rglob("*") if p.is_file()}
    assert rename.main(["acmerag", "--root", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "src/kbsdk -> src/acmerag".replace("/", "\\") in out.replace(
        "/", "\\"
    )
    assert {p: p.read_bytes() for p in repo.rglob("*") if p.is_file()} == before


def test_apply_renames_the_package_and_every_spelling_of_the_name(repo):
    assert rename.main(["acme_rag", "--root", str(repo), "--apply", "--force", "--no-tidy"]) == 0
    assert (repo / "src" / "acme_rag" / "errors.py").exists() and not (
        repo / "src" / "kbsdk"
    ).exists()
    assert (repo / "src/acme_rag/sub/acme_rag_meter.py").exists()  # files with the name in them too
    errors = read(repo, "src/acme_rag/errors.py")
    assert (
        "class AcmeRagError" in errors
        and '"acme_rag.plugins"' in errors
        and '".acme_rag"' in errors
    )
    assert 'os.environ["ACME_RAG_API_KEYS"]' in read(repo, "src/acme_rag/sub/acme_rag_meter.py")
    assert "from acme_rag.errors import AcmeRagError" in read(repo, "tests/test_x.py")
    pyproject = read(repo, "pyproject.toml")
    assert 'name = "acme_rag"' in pyproject and 'acme_rag = "acme_rag.cli:main"' in pyproject
    for text in (
        read(repo, "README.md"),
        read(repo, "pyproject.toml"),
        read(repo, "src/acme_rag/__init__.py"),
    ):
        assert "kbsdk" not in text.lower()


def test_the_placeholder_remarks_disappear_but_other_text_survives(repo):
    rename.main(["acmerag", "--root", str(repo), "--apply", "--force", "--no-tidy"])
    assert "placeholder" not in read(repo, "README.md") + read(repo, "pyproject.toml")
    assert "placeholder" not in read(repo, "src/acmerag/__init__.py")
    assert "pip install 'acmerag[server]'" in read(repo, "README.md")


def test_history_state_and_foreign_files_are_left_alone(repo):
    rename.main(["acmerag", "--root", str(repo), "--apply", "--force", "--no-tidy"])
    changelog = read(repo, "CHANGELOG.md")
    assert "Added kbsdk things." in changelog  # the past keeps the name it had
    assert "Renamed the package from `kbsdk` to `acmerag`" in changelog
    assert changelog.index("[Unreleased]") < changelog.index("[0.1.0]")
    assert (repo / ".kbsdk" / "index" / "manifest.json").exists()  # local state is not source
    assert read(repo, ".venv/lib/site.py") == "kbsdk = 1"
    assert "kbsdk" in read(repo, ".git/config")
    assert (repo / "blob.bin").read_bytes() == b"\xff\xfe\x00kbsdk\x80"  # binary untouched


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        ("Acme", "usable package name"),
        ("ab", "usable package name"),
        ("1acme", "usable package name"),
        ("acme-rag", "usable package name"),
        ("kbsdk", "same as the current"),
        ("json", "standard-library"),
        ("class", "keyword"),
    ],
)
def test_bad_names_are_rejected_before_anything_changes(repo, capsys, name, problem):
    assert rename.main([name, "--root", str(repo), "--apply", "--force", "--no-tidy"]) == 2
    assert problem in capsys.readouterr().err
    assert (repo / "src" / "kbsdk").exists()


def test_an_existing_target_package_is_a_collision(repo, capsys):
    (repo / "src" / "taken").mkdir()
    assert rename.main(["taken", "--root", str(repo), "--apply", "--force", "--no-tidy"]) == 2
    assert "already exists" in capsys.readouterr().err


def test_a_dirty_git_tree_is_refused_unless_forced(repo, capsys):
    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    (repo / "README.md").write_text("dirty", encoding="utf-8")
    assert rename.main(["acmerag", "--root", str(repo), "--apply"]) == 2
    assert "uncommitted" in capsys.readouterr().err and (repo / "src" / "kbsdk").exists()
    git("checkout", "--", "README.md")
    assert rename.main(["acmerag", "--root", str(repo), "--apply", "--no-tidy"]) == 0
    assert (repo / "src" / "acmerag").exists()


def test_mixed_case_spellings_are_reported(repo, capsys):
    (repo / "notes.md").write_text("The KbSdk project.", encoding="utf-8")
    rename.main(["acmerag", "--root", str(repo)])
    assert "mixed-case" in capsys.readouterr().out


def test_the_tool_and_its_test_keep_the_old_name_so_they_stay_valid(repo):
    (repo / "scripts").mkdir()
    (repo / "scripts" / "rename_package.py").write_text('OLD = "kbsdk"\n', encoding="utf-8")
    (repo / "tests" / "test_rename_script.py").write_text("# kbsdk fixtures\n", encoding="utf-8")
    rename.main(["acmerag", "--root", str(repo), "--apply", "--force", "--no-tidy"])
    assert read(repo, "scripts/rename_package.py") == 'OLD = "kbsdk"\n'
    assert read(repo, "tests/test_rename_script.py") == "# kbsdk fixtures\n"


def test_tidy_resorts_imports_when_ruff_is_available(repo, capsys):
    pytest.importorskip("ruff")
    (repo / "src" / "kbsdk" / "sorted_me.py").write_text(
        "from kbsdk import zebra\nfrom kbsdk import apple\nimport os\n", encoding="utf-8"
    )
    assert rename.main(["acmerag", "--root", str(repo), "--apply", "--force"]) == 0
    assert "re-sorted" in capsys.readouterr().out
    text = read(repo, "src/acmerag/sorted_me.py")
    assert text.index("import os") < text.index("from acmerag")
