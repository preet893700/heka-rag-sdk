# Releasing

The package is meant for a **private index** (Artifactory, Azure Artifacts, GitLab/GitHub Packages, devpi,
AWS CodeArtifact, ...). The metadata carries the `Private :: Do Not Upload` classifier, so `twine upload` to
the public PyPI is refused; a private index ignores it.

## 1. Pick the real name (once)

`kbsdk` is a placeholder. Before the first release:

```
python scripts/rename_package.py acmerag            # dry run: lists every edit
python scripts/rename_package.py acmerag --apply    # needs a clean git tree, so `git checkout .` undoes it
pip install -e .
python -m pytest && ruff check . && mypy
```

It renames the import package, the command, extras hints, the `<name>.plugins` entry-point group, the
environment-variable prefix (`KBSDK_JWT_SECRET` -> `ACMERAG_JWT_SECRET`), class names (`KbsdkError` ->
`AcmeragError`) and the default `.kbsdk/` state folder, then re-sorts imports with ruff. The changelog is left
as history and gets a note. Afterwards delete `scripts/rename_package.py` and `tests/unit/test_rename_script.py`.

Choose the name against your index, not just the file system: if a package with that name exists on the public
PyPI, an installer that falls back to it (`--extra-index-url`) could be tricked into installing the public one
(dependency confusion). Prefer a name that is unclaimed on PyPI, or register a placeholder there, and tell
consumers to use `--index-url` pointing only at your private index.

Also put your company's legal name in `LICENSE` (it currently says "the authors").

## 2. Cut a release

1. Move the `[Unreleased]` entries of `CHANGELOG.md` under a new version heading and set `version` in
   `pyproject.toml` (the only place: `<name>.__version__` reads it from the installed metadata).
2. Run the checks: `python -m pytest`, `ruff check .`, `ruff format --check .`, `mypy`.
3. Commit, then tag: `git tag -a v0.1.0 -m "0.1.0"`.
4. Build from a clean checkout of the tag:

   ```
   pip install build twine
   python -m build          # writes dist/<name>-<version>-py3-none-any.whl and .tar.gz
   twine check dist/*
   ```

5. Smoke-test the wheel in a fresh virtualenv (core dependencies only), before uploading:

   ```
   python -m venv /tmp/smoke && /tmp/smoke/bin/pip install dist/*.whl
   /tmp/smoke/bin/<name> presets
   /tmp/smoke/bin/python -c "import <name>; print(<name>.__version__)"
   ```

6. Upload to the private index. With `twine` (credentials in `~/.pypirc` or `TWINE_USERNAME` /
   `TWINE_PASSWORD`, never in the repository):

   ```
   twine upload --repository-url https://<your-index>/simple/ dist/*
   ```

   Azure Artifacts, CodeArtifact and Artifactory each document their own token step
   (`artifacts-keyring`, `aws codeartifact login --tool twine`, an API key); the upload command is the same.

## 3. Installing it

```
pip install --index-url https://<your-index>/simple/ "<name>[pdf,local,gemini]"
```

Extras are chosen per use: `pdf`, `ocr`, `office`, `web`, `local`, `gemini`, `groq`, `anthropic`, `openai`,
`ollama`, `qdrant`, `agentic`, `otel`, `server`, or `all`. A feature whose extra is missing fails with the exact
`pip install` command to run.

## What the build contains

* The wheel holds the package, its presets and `py.typed`; the sdist adds tests, examples, the README, licence
  and changelog. `.env` files, `.kbsdk/` index folders and `data/` are excluded from both.
* Versioning: `0.x` may change the public API in any minor release; from `1.0` the usual semantic versioning
  applies. Anything in `<name>.interfaces`, `RAGConfig`, `Agent`, `KnowledgeBase`, `Answer` and the eval
  dataset format is treated as public.
