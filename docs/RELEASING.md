# Releasing

[← back to README](../README.md)

The package is meant for a **private index** (Artifactory, Azure Artifacts, GitLab/GitHub Packages, devpi,
AWS CodeArtifact, ...). The metadata carries the `Private :: Do Not Upload` classifier, so `twine upload` to
the public PyPI is refused; a private index ignores it.

## 1. Names

| What | Name |
| ---- | ---- |
| Distribution (what you `pip install`) | `heka-rag-sdk` |
| Import | `heka.rag`  (`from heka.rag import Agent`) |
| Command line | `heka-rag` |
| Environment variables | `HEKA_RAG_*` (e.g. `HEKA_RAG_JWT_SECRET`) |
| Error base class | `HekaRagError` |
| Local state folder | `.heka-rag/` |
| Plug-in entry-point group | `heka.rag.plugins` |

`heka` is a **PEP 420 namespace package** shared by every Hekaos distribution (`heka.rag` now, `heka.agents` or
`heka.eval` later). The one rule: no distribution may ship a `heka/__init__.py`, or the namespace breaks for the
others. This repository keeps its code in `src/heka/rag/` and has no `src/heka/__init__.py`; the build checks in
section 2 verify that.

Choose names against your index, not just the file system: if a package named `heka-rag-sdk` exists on the public
PyPI, an installer that falls back to it (`--extra-index-url`) could be tricked into installing the public one
(dependency confusion). Register the name on the public PyPI as a placeholder, or tell consumers to use
`--index-url` pointing only at your private index.

`LICENSE` and the package metadata name **Hekaos** as copyright holder and author. If the legal entity needs a
suffix (Inc., Ltd., ...), change it in `LICENSE` and in `authors`/`maintainers` in `pyproject.toml`.

## 2. Cut a release

1. Move the `[Unreleased]` entries of `CHANGELOG.md` under a new version heading and set `version` in
   `pyproject.toml` (the only place: `heka.rag.__version__` reads it from the installed metadata).
2. Run the checks: `python -m pytest`, `ruff check .`, `ruff format --check .`, `mypy`.
3. Commit, then tag: `git tag -a v0.1.0 -m "0.1.0"`.
4. Build from a clean checkout of the tag:

   ```
   pip install build twine
   python -m build          # writes dist/heka_rag_sdk-<version>-py3-none-any.whl and .tar.gz
   twine check dist/*
   ```

5. Smoke-test the wheel in a fresh virtualenv (core dependencies only), before uploading:

   ```
   python -m venv /tmp/smoke && /tmp/smoke/bin/pip install dist/*.whl
   /tmp/smoke/bin/heka-rag presets
   /tmp/smoke/bin/python -c "import heka.rag as r; print(r.__version__, getattr(__import__('heka'), '__file__', None))"
   ```

   The last value must print `None`: `heka` is a namespace, so it has no `__file__`. Also confirm the wheel has no
   `heka/__init__.py`:

   ```
   python -c "import zipfile,glob; print('heka/__init__.py' in zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist())"   # False
   ```

6. Upload to the private index. With `twine` (credentials in `~/.pypirc` or `TWINE_USERNAME` /
   `TWINE_PASSWORD`, never in the repository):

   ```
   twine upload --repository-url https://<your-index>/simple/ dist/*
   ```

   Azure Artifacts, CodeArtifact and Artifactory each document their own token step
   (`artifacts-keyring`, `aws codeartifact login --tool twine`, an API key); the upload command is the same.

## 3. Installing it

**From a private package index** (after step 6 above; the recommended way for a team):

```
pip install --index-url https://<your-index>/simple/ "heka-rag-sdk[pdf,local,gemini]"
```

**Before an index exists, from the public GitHub repository** (`github.com/preet893700/heka-rag-sdk`). Both
commands were tested in clean environments with no GitHub login:

```
# from the git tag
pip install "heka-rag-sdk[pdf,local,gemini] @ git+https://github.com/preet893700/heka-rag-sdk.git@v0.1.0"

# from the wheel attached to the release (compare with the SHA-256 in the release notes)
pip install "heka-rag-sdk[pdf,local,gemini] @ https://github.com/preet893700/heka-rag-sdk/releases/download/v0.1.0/heka_rag_sdk-0.1.0-py3-none-any.whl"
```

The repository is public but the licence is proprietary (see `LICENSE`): visibility grants no right to use or
redistribute. If the repository is ever made private again, both commands stop working for people without
access: they would need a git login (or SSH key) for the first, and to download the wheel first, signed in, for
the second; a plain `pip install <release URL>` fails on private assets. Never put an access token in a
`pip install` URL (it is saved in shell history and logs).

Extras are chosen per use: `pdf`, `ocr`, `office`, `web`, `local`, `gemini`, `groq`, `anthropic`, `openai`,
`ollama`, `qdrant`, `agentic`, `otel`, `server`, or `all`. A feature whose extra is missing fails with the exact
`pip install` command to run.

## What the build contains

* The wheel holds the package, its presets and `py.typed`; the sdist adds tests, examples, the README, licence
  and changelog. `.env` files, `.heka-rag/` index folders and `data/` are excluded from both.
* Versioning: `0.x` may change the public API in any minor release; from `1.0` the usual semantic versioning
  applies. Anything in `heka.rag.interfaces`, `RAGConfig`, `Agent`, `KnowledgeBase`, `Answer` and the eval
  dataset format is treated as public.
