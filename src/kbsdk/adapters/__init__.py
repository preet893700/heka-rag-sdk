"""Built-in adapters. Importing this package registers them (lazily: nothing heavy is imported)."""

from __future__ import annotations

from kbsdk.registry import register_lazy

_A = "kbsdk.adapters"

# Loaders (file extensions are mapped in kbsdk.adapters.loaders.EXTENSION_LOADERS)
register_lazy("loader", "text", f"{_A}.loaders:TextLoader")
register_lazy("loader", "pdf", f"{_A}.loaders:PdfLoader", extra="pdf")
register_lazy("loader", "docx", f"{_A}.loaders:DocxLoader", extra="office")
register_lazy("loader", "pptx", f"{_A}.loaders:PptxLoader", extra="office")
register_lazy("loader", "xlsx", f"{_A}.loaders:XlsxLoader", extra="office")
register_lazy("loader", "csv", f"{_A}.loaders:CsvLoader")
register_lazy("loader", "html", f"{_A}.loaders:HtmlLoader", extra="web")

register_lazy("loader", "image", f"{_A}.loaders:ImageLoader", extra="ocr")
register_lazy("ocr", "rapidocr", f"{_A}.ocr:RapidOcr", extra="ocr")

register_lazy("chunker", "structure_aware", f"{_A}.chunkers:StructureAwareChunker")
register_lazy("chunker", "fixed", f"{_A}.chunkers:FixedChunker")
register_lazy("chunker", "parent_child", f"{_A}.chunkers_advanced:ParentChildChunker")
register_lazy("chunker", "semantic", f"{_A}.chunkers_advanced:SemanticChunker")

register_lazy("embedder", "hashing", f"{_A}.embedders:HashingEmbedder")
register_lazy("embedder", "local", f"{_A}.embedders:LocalEmbedder", extra="local")
register_lazy("embedder", "gemini", f"{_A}.embedders:GeminiEmbedder", extra="gemini")

register_lazy("store", "local", f"{_A}.stores:LocalVectorStore")
register_lazy("store", "qdrant", f"{_A}.qdrant_store:QdrantStore", extra="qdrant")

register_lazy("llm", "gemini", f"{_A}.llms:GeminiLLM", extra="gemini")
register_lazy("llm", "groq", f"{_A}.llms:GroqLLM", extra="groq")
register_lazy("llm", "claude", f"{_A}.llms:ClaudeLLM", extra="anthropic")
register_lazy("llm", "openai", f"{_A}.llms:OpenAILLM", extra="openai")
register_lazy("llm", "ollama", f"{_A}.llms:OllamaLLM", extra="ollama")

register_lazy("embedder", "openai", f"{_A}.embedders:OpenAIEmbedder", extra="openai")
register_lazy("embedder", "ollama", f"{_A}.embedders:OllamaEmbedder", extra="ollama")

register_lazy("cache", "disk", f"{_A}.caches:DiskCache")
register_lazy("cache", "memory", f"{_A}.caches:MemoryCache")

register_lazy("tracer", "memory", f"{_A}.tracers:MemoryTracer")
register_lazy("tracer", "logging", f"{_A}.tracers:LoggingTracer")
register_lazy("tracer", "jsonl", f"{_A}.tracers:JsonlTracer")
register_lazy("tracer", "otel", f"{_A}.tracers:OtelTracer", extra="otel")

register_lazy("guardrail", "pii", f"{_A}.guardrails:PiiGuardrail")
register_lazy("guardrail", "injection", f"{_A}.guardrails:InjectionGuardrail")
register_lazy("guardrail", "scope", f"{_A}.guardrails:ScopeGuardrail")

register_lazy("reranker", "local_cross_encoder", f"{_A}.rerankers:LocalCrossEncoder", extra="local")
register_lazy("reranker", "llm", f"{_A}.rerankers:LLMReranker")

register_lazy("query_transform", "rewrite", f"{_A}.transforms:RewriteTransform")
register_lazy("query_transform", "multi_query", f"{_A}.transforms:MultiQueryTransform")
register_lazy("query_transform", "hyde", f"{_A}.transforms:HydeTransform")
