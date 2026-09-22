# Images

[← back to README](../README.md)

## What's in place

* **`docs/images/architecture.png`** — the full architecture diagram used under "## Architecture" in the
  README (document inputs → SDK core → pluggable adapters → real providers → output, plus the three access
  points: Python API, CLI, REST server). Supplied directly, not generated from the prompt below; the prompt
  is kept in case it ever needs regenerating or updating for a new provider/stage.
* **`docs/images/heka-rag-sdk-lockup.png`** — the single combined image used at the very top of the README
  (icon + wordmark, centered). Composited from two source files (below) rather than referenced as two
  separate images: this app's file-pane preview renders each Markdown image as its own block, so two
  `![]()` images on the same source line still stacked one above the other instead of sitting side by side.
  One pre-built image sidesteps that everywhere, GitHub included, with no reliance on how any particular
  renderer handles adjacent inline images. It's wrapped in a one-cell, center-aligned Markdown table (`|:-:|`)
  in the README to center it — the pure-Markdown equivalent of `<p align="center">`, since raw HTML doesn't
  render in this app's preview either.
* **`docs/images/heka-rag-sdk-wordmark.png`** and **`docs/images/hekaos-icon.png`** (72×74) — the two source
  images the lockup above is composited from (wordmark crop + Hekaos icon, background keyed to transparent
  where needed). Not referenced directly in the README any more, kept as the editable components.
* **`docs/images/hekaos-footer-credit.png`** — the small "Built by Hekaos." credit at the very end of the
  README. Same reasoning as the lockup: an icon image followed by plain text on the same line hit the same
  block-stacking problem, so this is one composited image (icon + rendered text, Segoe UI Semibold, coloured
  to match the wordmark's tagline) rather than an image plus a separate text node.
* **`docs/images/hekaos-icon-small.png`** (20×20) — a smaller export of the icon; the source the footer
  credit's icon half is composited from.
* **`docs/images/flow.png`** and **`docs/images/pipeline.png`** — the two diagrams under "## See it work"
  and "### The pipeline, one level deeper". Supplied directly, in the same custom illustrated style as
  `architecture.png` (built from the prompts below). Both used to be live ` ```mermaid ` code fences —
  accurate, and GitHub renders them natively, but a raw Mermaid block shows as plain code text (not a
  diagram) in any viewer that doesn't support it, this app's own file pane among them — then briefly
  Mermaid rendered to PNG (a generic look) before these were supplied.

The architecture diagram and wordmark came from one source image (a full architecture infographic with the
wordmark built into its header), split apart early on. The icon is a separate, smaller source. `flow.png`
and `pipeline.png` are two further supplied images, matching that same palette.

**Every image is referenced with plain Markdown syntax** (`![alt](docs/images/....png)`), not HTML `<img>`
tags. The README briefly used HTML tags to get explicit `width=`/`style=` control, but this app's own
file-pane preview doesn't render raw HTML in Markdown at all — it shows the tag as literal text, not even a
broken-image icon. Plain Markdown image syntax renders everywhere (GitHub, this preview, PyPI, ...), which
is why sizing and layout (the lockup composite, the centering table) are done by pre-building the actual
file/structure rather than by attribute: the portable way to get a specific look when it has to work in
every renderer, not just ones that allow HTML.

### Rebuilding the lockup image

If the icon or wordmark changes, rebuild `docs/images/heka-rag-sdk-lockup.png` from its two sources
(`docs/images/hekaos-icon.png`, `docs/images/heka-rag-sdk-wordmark.png`) with a short Pillow script: key the
wordmark's near-white background to transparent (`min(r,g,b) >= 250` → alpha 0, with a short ramp down to
`230` for anti-aliased edges), then paste both onto one transparent canvas, icon on the left, an ~18px gap,
wordmark on the right, both vertically centered.

### Rebuilding the footer credit image

Same idea, smaller: paste `docs/images/hekaos-icon-small.png` onto a transparent canvas, then use
`PIL.ImageDraw` with `C:\Windows\Fonts\seguisb.ttf` at 20pt, colour `(100, 111, 145)` (sampled from the
wordmark's tagline), to draw "Built by Hekaos." beside it with an 8px gap, both vertically centered. Update
the text or swap the font there if the credit line's wording changes.

A now-unused fallback still sits in `docs/images/src/*.mmd` (the plain-Mermaid version of the flow/pipeline
diagrams) in case a quick, ungenerated regeneration of either is ever needed — `npx --yes
@mermaid-js/mermaid-cli -i docs/images/src/flow.mmd -o docs/images/flow.png -b white -s 3` (swap in
`pipeline.mmd`/`pipeline.png` for the other one). Not needed while the supplied images above are in place.

**Worth knowing:** `D:\hekaos\public\logo192.png` and `logo512.png` are *not* Hekaos assets — they're the
default React-atom logo that `create-react-app` scaffolds into every new project and that nobody replaced.
`favicon.ico` in that folder is the real Hekaos mark too, just at a much lower resolution (48×48 max) than
`docs/images/src/hekaos-icon-source.png`, which is what the README's icon is actually built from now.

## The architecture prompt (for regenerating or updating it later)

Written for ChatGPT's image generation (DALL·E). Useful if the provider list changes, a new pipeline stage
ships, or the image needs a refresh — paste it in, save the result over `docs/images/architecture.png`.

> Create a clean, modern technical architecture diagram (flat vector infographic style, not a screenshot and
> not photorealistic) for a software SDK called "heka-rag-sdk", laid out in labelled horizontal layers on a
> white background, landscape orientation, about 1600×900px, suitable for embedding at full width in a
> GitHub README:
>
> **Layer 1 (input, left):** small labelled icons for the document types it reads — PDF, Word, PowerPoint,
> Excel, CSV, HTML, a scanned image — plus a small config-file icon, all flowing right into layer 2.
>
> **Layer 2 (core, center-left):** one prominent rounded box labelled "heka-rag-sdk core — KnowledgeBase +
> Agent", subtitled "stable Python API".
>
> **Layer 3 (pluggable adapters, center):** a row of smaller labelled boxes connected to the core box,
> styled like plug/socket or puzzle-piece connectors to suggest "swappable": "Loaders", "Chunkers",
> "Embedders", "Vector Store", "Retriever / Reranker", "Guardrails", "LLM", "Tracer / Cache".
>
> **Layer 4 (real providers, center-right):** small labelled chips grouped under the adapter box they plug
> into — "Gemini, Groq, Claude, OpenAI, Ollama" under LLM; "Local store, Qdrant" under Vector Store; "Local,
> Gemini, OpenAI, Ollama" under Embedders.
>
> **Layer 5 (output, right):** an arrow leading to a result card labelled "Answer + verified citations +
> usage + trace", with three small icons beneath it for how it's reached: "Python API", "CLI", "REST
> server".
>
> Style: flat 2D vector infographic, rounded rectangles, thin arrowed connecting lines, a restrained 3-4
> colour palette (deep indigo/violet for the core box, teal or blue for the adapter layer, neutral grey for
> external providers, white background), clean sans-serif labels, generous whitespace, no drop shadows, no
> gradients, no photorealistic elements — the visual style of a modern developer-tool architecture diagram
> (similar to Vercel, Supabase, Temporal or LangChain's own documentation diagrams). Text must stay legible
> when the image is scaled down to fit a narrow screen.

## The flow-diagram prompt (for regenerating or updating `flow.png` later)

Kept for updating the image later (a new step, a wording change) rather than editing it by hand — paste in,
save the result over `docs/images/flow.png`:

> Create a clean, modern flat vector infographic (not a screenshot, not photorealistic) showing a simple
> left-to-right data flow for a software SDK called "heka-rag-sdk", on a white background, landscape
> orientation, about 1600×500px, suitable for embedding at full width in a GitHub README, in the same visual
> style as heka-rag-sdk's architecture diagram: rounded-rectangle boxes with deep indigo/violet borders
> (~#3B4A9E) and light-blue fills (~#EAF1FE), teal accents (~#2F9E90 border / #CBF5EF fill) for secondary
> elements, thin arrowed connecting lines, clean sans-serif labels, generous whitespace, no drop shadows, no
> gradients:
>
> **Left side, two small labelled icon boxes stacked vertically**, arrows converging to the right: a
> document-stack icon labelled "Your documents", and a gear/config icon labelled "Your config".
>
> **Center-left:** those arrows join into a rounded box labelled "KnowledgeBase.ingest()".
>
> **Center:** an arrow from that box to a small cylinder/database icon labelled "Index".
>
> **Center-right, a third input joining from below:** a question-mark icon box labelled "A question", its
> arrow joining the arrow coming from "Index".
>
> **Right side:** those arrows lead into a rounded box labelled "Agent.ask()", with a final arrow to a
> highlighted result card (teal-accented, like the architecture diagram's output card) with a green
> checkmark icon, labelled "Answer + checked citations".
>
> Style: exactly matching the heka-rag-sdk architecture diagram's palette and typography, so the two look
> like they belong to the same document. Text must stay legible when the image is scaled down to fit a
> narrow screen.

## The pipeline-diagram prompt (for regenerating or updating `pipeline.png` later)

Kept for updating the image later (a new pipeline step, a wording change) rather than editing it by hand —
paste in, save the result over `docs/images/pipeline.png`:

> Create a clean, modern flat vector infographic (not a screenshot, not photorealistic) showing a
> left-to-right pipeline flowchart for a software SDK called "heka-rag-sdk", on a white background,
> landscape orientation, wide format (about 2400×500px — this chain has 14 steps and needs the room),
> suitable for embedding at full width in a GitHub README, in the same visual style as heka-rag-sdk's
> architecture diagram: rounded-rectangle boxes with deep indigo/violet borders (~#3B4A9E) and light-blue
> fills (~#EAF1FE), teal accents (~#2F9E90 border / #CBF5EF fill) for the decision/branch elements, thin
> arrowed connecting lines, clean sans-serif labels, no drop shadows, no gradients:
>
> A single horizontal chain of labelled rounded boxes, left to right, connected by arrows, in this exact
> order: "Question" → "Input guardrails" → "Condense follow-up" → "Rewrite / multi-query / HyDE" →
> "Retrieve (access-filtered)" → "Fuse results" → "Rerank" → "Expand to parent section" → "Context
> guardrails" → a diamond-shaped decision box labelled "Enough grounding?".
>
> From the decision diamond, two branches: one labelled "no" leading to a box "Decline, with a reason" (a
> dead-end branch, styled slightly muted/grey to show it exits here); one labelled "yes" continuing the main
> chain into "Answer + verified citations" → "Answer verification" → "Output guardrails" → "Final answer"
> (this last box styled as a highlighted teal-accented result card, like the architecture diagram's output
> card).
>
> Style: exactly matching the heka-rag-sdk architecture diagram's palette and typography, so all the
> README's diagrams look like they belong to the same document. Text must stay legible at this wide, thin
> aspect ratio when scaled to fit a narrow screen — stack each label onto two lines inside its box if one
> line would run too wide.

## Regenerating the icon at other sizes

No prompt needed — `docs/images/src/hekaos-icon-source.png` (494×505, transparent) is high enough resolution
to downsize for any use. To make another size:

```bash
python -c "
from PIL import Image
src = Image.open('docs/images/src/hekaos-icon-source.png').convert('RGBA')
w = 96  # target width in px
h = round(w * src.height / src.width)
src.resize((w, h), Image.LANCZOS).save('docs/images/hekaos-icon-<size>.png')
"
```

## Other images: not added, only if you want them

Nothing else in the README needed an image to make its point. A "verified citations" or "before/after"
graphic could work for the **Why heka-rag-sdk?** section, but the plain-language pain-point list there
already carries that weight in text. Ask for a prompt for a specific section if you want one.
