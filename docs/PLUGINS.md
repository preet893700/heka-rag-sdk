# Plug-ins

[← back to README](../README.md)

Every stage is a small `Protocol` in `heka.rag.interfaces`. Register an adapter and use it by name in any
config; runtime collaborators (an LLM, an embedder) are injected only if the constructor asks for them:

```python
from heka.rag import register
from heka.rag.adapters._deps import Settings


class MyGuardrailSettings(Settings):  # validated; unknown keys are errors
    banned: list[str] = []


@register("guardrail", "banned_words")
class BannedWords:
    settings_model = MyGuardrailSettings

    def __init__(self, settings):
        self.banned = settings.banned

    async def check(self, text, *, stage, context=None):
        from heka.rag.types import GuardrailResult

        hit = next((w for w in self.banned if w in text.lower()), None)
        return (
            GuardrailResult(action="block", reason=hit, message="Please rephrase.")
            if hit
            else GuardrailResult()
        )
```

Use it from any config exactly like a built-in provider:

```yaml
guardrails:
  - provider: banned_words
    params: {banned: ["confidential"]}
```

Third-party packages can ship adapters through the `heka.rag.plugins` entry-point group, so a company-wide
loader, store or guardrail can be `pip install`-ed rather than copy-pasted between projects.
