from kbsdk import Answer, Chunk, Citation, ScoredChunk, Usage
from kbsdk.types import LLMResponse


def test_usage_addition_sums_and_handles_missing_cost():
    a = Usage(input_tokens=10, output_tokens=5, llm_calls=1)
    b = Usage(input_tokens=1, output_tokens=2, llm_calls=1, cost_usd=0.5)
    total = a + b
    assert (total.input_tokens, total.output_tokens, total.llm_calls) == (11, 7, 2)
    assert total.cost_usd == 0.5
    assert (a + a).cost_usd is None


def test_answer_sources_are_distinct_and_ordered():
    answer = Answer(
        text="x",
        citations=[
            Citation(source="b.pdf", chunk_id="1", quote="q"),
            Citation(source="a.pdf", chunk_id="2", quote="q"),
            Citation(source="b.pdf", chunk_id="3", quote="q"),
        ],
    )
    assert answer.sources == ["b.pdf", "a.pdf"]


def test_answer_round_trips_through_json():
    answer = Answer(
        text="12 days.",
        citations=[Citation(source="leave.pdf", chunk_id="c1", quote="12 days", verified=True)],
        retrieved=[ScoredChunk(chunk=Chunk(id="c1", doc_id="d", text="12 days"), score=0.9)],
        abstained=False,
        confidence=0.8,
    )
    assert Answer.model_validate_json(answer.model_dump_json()) == answer


def test_llm_response_never_serializes_raw_payload():
    response = LLMResponse(text="hi", raw={"secret": "provider payload"})
    assert "raw" not in response.model_dump()
