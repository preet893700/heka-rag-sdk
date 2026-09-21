"""Rename the placeholder package (`kbsdk`) to its real name across the whole repository.

    python scripts/rename_package.py acmerag            # dry run: shows what would change
    python scripts/rename_package.py acmerag --apply    # does it

Everything that carries the name is rewritten together, so nothing is left half-renamed:
  * the import package (`src/kbsdk` -> `src/<name>`) and every `import kbsdk` / `kbsdk.x` reference,
  * the distribution and command names, extras hints (`pip install '<name>[server]'`) and the
    `<name>.plugins` entry-point group,
  * the environment-variable prefix (`KBSDK_JWT_SECRET` -> `<NAME>_JWT_SECRET`),
  * class names (`KbsdkError` -> `<Name>Error`) and the default `.kbsdk/` state folder.
`CHANGELOG.md` is history and is left alone; a note about the rename is added to it instead.

It refuses to run on a dirty git tree (so `git checkout .` undoes it) unless `--force` is given.
Afterwards: `pip install -e .`, run the tests, and re-create any `.kbsdk/` index folders (they are
rebuilt from your documents by `<name> ingest`; the old folders are not renamed).
"""

from __future__ import annotations

import argparse
import keyword
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

OLD = "kbsdk"
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
SKIP_DIR_SUFFIXES = (".egg-info",)
# History, plus this tool and its test (they only make sense while the name is still the old one).
SKIP_FILES = {"CHANGELOG.md", "rename_package.py", "test_rename_script.py"}
MAX_BYTES = 5_000_000
PATTERN = re.compile(OLD, re.IGNORECASE)


class RenameError(Exception):
    pass


def validate_name(new: str, root: Path) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,}", new):
        raise RenameError(
            f"{new!r} is not a usable package name: use lowercase letters, digits and underscores, "
            "starting with a letter, at least 3 characters (e.g. acmerag)."
        )
    if new == OLD:
        raise RenameError(f"The new name is the same as the current one ({OLD!r}).")
    if keyword.iskeyword(new) or new in sys.stdlib_module_names:
        raise RenameError(f"{new!r} clashes with a Python keyword or standard-library module.")
    if (root / "src" / new).exists():
        raise RenameError(f"src/{new} already exists.")


def camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def replace_text(text: str, new: str) -> tuple[str, int, list[str]]:
    """Case-aware replacement. Returns the new text, the number of replacements and odd spellings."""
    odd: list[str] = []
    count = 0

    def sub(match: re.Match[str]) -> str:
        nonlocal count
        found = match.group(0)
        count += 1
        if found == OLD:
            return new
        if found == OLD.upper():
            return new.upper()
        if found == OLD.capitalize():
            return camel(new)
        odd.append(found)
        return new  # mixed case: best effort, reported

    replaced = PATTERN.sub(sub, text)
    return replaced, count, odd


def cleanup_placeholder_remarks(text: str, new: str) -> str:
    """Drop the sentences that said the name was a placeholder; they are no longer true."""
    for stale in (
        "  # placeholder name - rename before the first release",
        " (placeholder name)",
        f" `{new}` is a placeholder name.",
        f"`{new}` is a placeholder name. ",
    ):
        text = text.replace(stale, "")
    return text


@dataclass
class FileEdit:
    path: Path
    replacements: int
    new_text: str
    odd: list[str] = field(default_factory=list)


@dataclass
class Plan:
    root: Path
    new: str
    edits: list[FileEdit] = field(default_factory=list)
    renames: list[tuple[Path, Path]] = field(default_factory=list)
    unreadable: list[Path] = field(default_factory=list)

    @property
    def total_replacements(self) -> int:
        return sum(e.replacements for e in self.edits)


def _walk(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        parts = rel.parts
        if any(p in SKIP_DIRS or p.endswith(SKIP_DIR_SUFFIXES) for p in parts):
            continue
        if any(p == f".{OLD}" for p in parts):  # local index/state folders: data, not source
            continue
        yield path


def make_plan(root: Path, new: str) -> Plan:
    validate_name(new, root)
    plan = Plan(root=root, new=new)
    for path in _walk(root):
        if path.is_dir():
            if OLD in path.name.lower():
                plan.renames.append((path, path.with_name(PATTERN.sub(new, path.name))))
            continue
        if path.name in SKIP_FILES or path.stat().st_size > MAX_BYTES:
            continue
        if OLD in path.name.lower():
            plan.renames.append((path, path.with_name(PATTERN.sub(new, path.name))))
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            plan.unreadable.append(path)  # binary: left untouched
            continue
        if OLD not in text.lower():
            continue
        replaced, count, odd = replace_text(text, new)
        plan.edits.append(FileEdit(path, count, cleanup_placeholder_remarks(replaced, new), odd))
    plan.renames.sort(key=lambda pair: len(pair[0].parts), reverse=True)  # children before parents
    return plan


def append_changelog_note(root: Path, new: str) -> bool:
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        return False
    text = changelog.read_text(encoding="utf-8")
    note = (
        f"## [Unreleased]\n\n### Changed\n- Renamed the package from `{OLD}` to `{new}`: import name, command,\n"
        f"  extras, entry-point group (`{new}.plugins`) and environment-variable prefix "
        f"(`{OLD.upper()}_*` -> `{new.upper()}_*`).\n\n"
    )
    marker = "## ["
    index = text.find(marker)
    changelog.write_text(text[:index] + note + text[index:] if index >= 0 else text + "\n" + note)
    return True


def apply_plan(plan: Plan) -> None:
    for edit in plan.edits:
        edit.path.write_bytes(edit.new_text.encode("utf-8"))
    for source, target in plan.renames:
        if target.exists():
            raise RenameError(f"cannot rename {source} -> {target}: the target exists")
        source.rename(target)
    append_changelog_note(plan.root, plan.new)


def git_is_dirty(root: Path) -> bool | None:
    """True/False, or None when this is not a git repository (or git is missing)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def tidy(root: Path) -> str:
    """Re-sort imports and re-wrap lines with ruff, since the new name sorts and measures differently."""
    steps = (
        ["check", "--select", "I", "--fix", "-q", str(root)],
        ["format", "-q", str(root)],
    )
    for step in steps:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "ruff", *step], capture_output=True, text=True
            )
        except OSError as exc:
            return f"skipped tidying imports/format ({exc}); run: ruff check --select I --fix . && ruff format ."
        if "No module named ruff" in result.stderr:
            return "ruff is not installed: run `ruff check --select I --fix . && ruff format .` yourself."
    return "imports re-sorted and code re-formatted with ruff."


def describe(plan: Plan) -> str:
    lines = [
        f"rename {OLD!r} -> {plan.new!r}: {plan.total_replacements} replacements in "
        f"{len(plan.edits)} files, {len(plan.renames)} paths renamed"
    ]
    for source, target in plan.renames:
        lines.append(
            f"  rename  {source.relative_to(plan.root)} -> {target.relative_to(plan.root)}"
        )
    for edit in plan.edits:
        lines.append(f"  edit    {edit.path.relative_to(plan.root)}  ({edit.replacements})")
        if edit.odd:
            lines.append(
                f"          note: mixed-case spellings rewritten as lowercase: {sorted(set(edit.odd))}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("name", help="the new package name (lowercase, e.g. acmerag)")
    parser.add_argument("--apply", action="store_true", help="make the changes (default: dry run)")
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    parser.add_argument("--force", action="store_true", help="allow a dirty git tree")
    parser.add_argument(
        "--no-tidy", action="store_true", help="do not run ruff to re-sort imports afterwards"
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        plan = make_plan(root, args.name)
        print(describe(plan))
        if not args.apply:
            print("\ndry run: nothing changed. Re-run with --apply to make these changes.")
            return 0
        dirty = git_is_dirty(root)
        if dirty and not args.force:
            raise RenameError(
                "the git working tree has uncommitted changes; commit or stash them first "
                "(so the rename can be undone with git), or pass --force."
            )
        apply_plan(plan)
        tidied = None if args.no_tidy else tidy(root)
    except RenameError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if tidied:
        print(tidied)
    print(
        f"\ndone. Next: pip install -e .   then run the tests. Old .{OLD}/ index folders are not "
        f"renamed; `{args.name} ingest` rebuilds them. scripts/rename_package.py and its test only "
        "make sense while the name is the old one: delete them once you are done."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
