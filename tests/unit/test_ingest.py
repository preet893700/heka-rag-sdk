import pytest

from kbsdk import ConfigError, KnowledgeBase
from kbsdk.pipelines.ingest import discover


async def test_first_ingest_indexes_everything(make_config):
    kb = KnowledgeBase(make_config())
    report = await kb.aingest()
    assert (report.files_seen, report.files_new, report.files_unchanged) == (2, 2, 0)
    assert report.chunks_added == report.chunks_total > 0
    assert not report.failed and not report.rebuilt
    assert "2 seen" in report.summary()


async def test_second_ingest_does_no_work(make_config):
    kb = KnowledgeBase(make_config())
    await kb.aingest()
    report = await kb.aingest()
    assert (report.files_unchanged, report.files_new, report.files_changed) == (2, 0, 0)
    assert report.chunks_added == 0


async def test_changed_new_and_removed_files(make_config, docs_dir):
    kb = KnowledgeBase(make_config())
    await kb.aingest()
    before = await kb.acount()

    (docs_dir / "leave.md").write_text(
        "# Leave Policy\n\n## Casual leave\n\nNow 15 days.\n", encoding="utf-8"
    )
    (docs_dir / "new.md").write_text("# Benefits\n\nHealth cover is included.\n", encoding="utf-8")
    (docs_dir / "expenses.md").unlink()
    report = await kb.aingest()

    assert (report.files_changed, report.files_new, report.files_removed) == (1, 1, 1)
    assert report.chunks_removed > 0
    sources = {c.chunk.metadata["source"] for c in await kb.aretrieve("leave", k=50)}
    assert sources == {"leave.md", "new.md"}  # the removed file is gone from the index
    texts = " ".join(c.chunk.text for c in await kb.aretrieve("casual leave", k=50))
    assert "Now 15 days." in texts and "12 days" not in texts  # the old version is gone
    assert await kb.acount() != before


async def test_changing_the_embedder_forces_a_rebuild(make_config):
    kb = KnowledgeBase(make_config())
    await kb.aingest()
    other = KnowledgeBase(
        make_config(embedder={"provider": "hashing", "params": {"dimensions": 128}})
    )
    report = await other.aingest()
    assert report.rebuilt
    assert report.files_new == 2  # everything re-processed, no dimension-mismatch crash
    assert len(await other.aretrieve("leave", k=1)) == 1


async def test_rebuild_mode_always_rebuilds(make_config):
    config = make_config(
        knowledge={**make_config().to_dict()["knowledge"], "update_mode": "rebuild"}
    )
    kb = KnowledgeBase(config)
    await kb.aingest()
    assert (await kb.aingest()).rebuilt


async def test_lost_index_with_surviving_manifest_is_repaired(make_config):
    kb = KnowledgeBase(make_config())
    await kb.aingest()
    await kb.store.clear()  # index wiped, manifest still says everything is indexed
    report = await kb.aingest()
    assert report.rebuilt and await kb.acount() > 0


async def test_source_metadata_reaches_chunks_and_changes_trigger_reindex(make_config, docs_dir):
    base = make_config().to_dict()
    tagged = {
        **base["knowledge"],
        "sources": [{"location": str(docs_dir), "metadata": {"department": "hr"}}],
    }
    kb = KnowledgeBase(make_config(knowledge=tagged))
    await kb.aingest()
    assert all(r.chunk.metadata["department"] == "hr" for r in await kb.aretrieve("leave", k=10))
    retagged = {
        **tagged,
        "sources": [{"location": str(docs_dir), "metadata": {"department": "legal"}}],
    }
    report = await KnowledgeBase(make_config(knowledge=retagged)).aingest()
    assert report.files_changed == 2


async def test_one_bad_file_does_not_abort_the_run(make_config, docs_dir):
    (docs_dir / "broken.docx").write_bytes(b"this is not a real docx")
    kb = KnowledgeBase(make_config())
    report = await kb.aingest()
    assert [f.path for f in report.failed] == ["broken.docx"]
    assert report.files_new == 3
    assert await kb.acount() > 0
    assert "FAILED broken.docx" in report.summary()
    # the failed file is retried next time rather than remembered as done
    assert [f.path for f in (await kb.aingest()).failed] == ["broken.docx"]


async def test_unsupported_and_hidden_files_are_skipped(make_config, docs_dir):
    (docs_dir / "image.png").write_bytes(b"\x89PNG")
    (docs_dir / ".hidden.md").write_text("# Secret", encoding="utf-8")
    (docs_dir / "~$lock.docx").write_bytes(b"x")
    (docs_dir / ".git").mkdir()
    (docs_dir / ".git" / "config.md").write_text("# no", encoding="utf-8")
    report = await KnowledgeBase(make_config()).aingest()
    assert report.files_seen == 2
    assert any(
        "image file" in w and "image.png" in w and "knowledge.ocr" in w for w in report.warnings
    )
    (docs_dir / "data.bin").write_bytes(b"\x00")
    report = await KnowledgeBase(make_config()).aingest()
    assert any("unsupported" in w and "data.bin" in w for w in report.warnings)


async def test_empty_files_produce_a_warning(make_config, docs_dir):
    (docs_dir / "empty.md").write_text("   \n", encoding="utf-8")
    report = await KnowledgeBase(make_config()).aingest()
    assert any("empty.md" in w and "no text" in w for w in report.warnings)


async def test_scanned_pdf_warns_about_ocr(make_config, docs_dir):
    pytest.importorskip("pypdf")
    pytest.importorskip("pdfplumber")
    from pdf_helpers import make_pdf

    (docs_dir / "scan.pdf").write_bytes(make_pdf(["Some text on page one, plenty of it.", ""]))
    report = await KnowledgeBase(make_config()).aingest()
    assert any("scan.pdf" in w and "knowledge.ocr" in w for w in report.warnings)


async def test_ocr_pipeline_end_to_end_reads_scans_and_images(make_config, docs_dir):
    pytest.importorskip("pdfplumber")
    pil = pytest.importorskip("PIL.Image")
    from kbsdk import registry
    from pdf_helpers import make_pdf

    class Ocr:
        async def recognize(self, image):
            return "# Notice\n\nThe notice period is thirty days."

    registry.register("ocr", "fake_for_ingest")(Ocr)
    (docs_dir / "scan.pdf").write_bytes(make_pdf(["A page with text, plenty of it here.", ""]))
    pil.new("RGB", (100, 60), "white").save(docs_dir / "photo.png")
    config = make_config(
        knowledge={**make_config().to_dict()["knowledge"], "ocr": {"provider": "fake_for_ingest"}}
    )
    kb = KnowledgeBase(config)
    report = await kb.aingest()
    assert not report.failed
    sources = {
        c.chunk.metadata["source"] for c in await kb.aretrieve("notice period thirty days", k=10)
    }
    assert {"scan.pdf", "photo.png"} <= sources
    assert any("scan.pdf" in w and "read with OCR" in w for w in report.warnings)
    # turning OCR on changes what is extracted, so an index built without it must be rebuilt
    assert (await KnowledgeBase(make_config()).aingest()).rebuilt


def test_discover_glob_and_single_file(docs_dir):
    from kbsdk.config import SourceConfig

    files, _ = discover([SourceConfig(location=str(docs_dir / "*.md"))])
    assert sorted(f.rel for f in files) == ["expenses.md", "leave.md"]
    files, _ = discover([SourceConfig(location=str(docs_dir / "leave.md"))])
    assert [f.rel for f in files] == ["leave.md"]


def test_discover_same_name_in_two_sources_stays_distinct(tmp_path):
    from kbsdk.config import SourceConfig

    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "policy.md").write_text("# P", encoding="utf-8")
    files, _ = discover(
        [SourceConfig(location=str(tmp_path / "a")), SourceConfig(location=str(tmp_path / "b"))]
    )
    assert len({f.rel for f in files}) == 2


def test_a_file_reached_through_two_sources_with_different_tags_is_an_error(docs_dir):
    """Otherwise a restricted document could be indexed a second time under public tags."""
    from kbsdk.config import SourceConfig

    broad = SourceConfig(location=str(docs_dir), metadata={"allowed_roles": ["*"]})
    narrow = SourceConfig(location=str(docs_dir / "leave.md"), metadata={"allowed_roles": ["hr"]})
    with pytest.raises(ConfigError, match="two sources with different settings"):
        discover([broad, narrow])
    with pytest.raises(ConfigError, match="exclude"):
        discover([narrow, broad])  # in either order


def test_identical_overlapping_sources_are_merged_not_duplicated(docs_dir):
    from kbsdk.config import SourceConfig

    same = {"allowed_roles": ["*"]}
    files, _ = discover(
        [
            SourceConfig(location=str(docs_dir), metadata=same),
            SourceConfig(location=str(docs_dir / "leave.md"), metadata=same),
        ]
    )
    assert sorted(f.rel for f in files) == ["expenses.md", "leave.md"]


def test_exclude_carves_a_file_out_of_a_broader_source(docs_dir):
    from kbsdk.config import SourceConfig

    (docs_dir / "drafts").mkdir()
    (docs_dir / "drafts" / "wip.md").write_text("# WIP", encoding="utf-8")
    broad = SourceConfig(
        location=str(docs_dir), exclude=["leave.md", "drafts/*"], metadata={"r": "*"}
    )
    narrow = SourceConfig(location=str(docs_dir / "leave.md"), metadata={"r": "hr"})
    files, _ = discover([broad, narrow])  # no conflict any more
    by_rel = {f.rel: f.source.metadata["r"] for f in files}
    assert by_rel == {"expenses.md": "*", "leave.md": "hr"}


async def test_no_document_can_be_indexed_under_two_tag_sets(make_config, docs_dir):
    from kbsdk import RequestContext

    knowledge = make_config().to_dict()["knowledge"]
    knowledge["sources"] = [
        {"location": str(docs_dir), "exclude": ["expenses.md"], "metadata": {"roles": ["*"]}},
        {"location": str(docs_dir / "expenses.md"), "metadata": {"roles": ["finance"]}},
    ]
    kb = KnowledgeBase(make_config(knowledge=knowledge, access={"roles_field": "roles"}))
    await kb.aingest()
    seen = {
        c.chunk.metadata["source"]
        for c in await kb.aretrieve(
            "expense claims", k=20, context=RequestContext(roles=["employee"])
        )
    }
    assert "expenses.md" not in seen and "leave.md" in seen


def test_discover_rejects_missing_paths_and_urls():
    from kbsdk.config import SourceConfig

    with pytest.raises(ConfigError, match="not found"):
        discover([SourceConfig(location="/definitely/not/here")])
    with pytest.raises(ConfigError, match="not available yet"):
        discover([SourceConfig(location="https://example.com/handbook")])
