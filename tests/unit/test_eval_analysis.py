import pytest

from kbsdk import Answer
from kbsdk.eval.analysis import (
    ANSWERED_UNANSWERABLE,
    CITATION_PROBLEM,
    ERROR,
    OK,
    RETRIEVAL_MISS,
    WRONG_ABSTENTION,
    WRONG_WITH_RIGHT_CONTEXT,
    by_tag,
    interval_of,
    metric_intervals,
    outcome_breakdown,
    outcome_of,
    wilson_interval,
)
from kbsdk.eval.report import CaseResult, EvalReport, aggregate


def case(case_id="c", answerable=True, tags=(), error=None, **metrics):
    return CaseResult(
        case_id=case_id,
        question=f"question {case_id}",
        split="dev",
        tags=list(tags),
        answerable=answerable,
        error=error,
        answer=Answer(text="an answer"),
        metrics=metrics,
    )


# -- intervals -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("successes", "n", "low", "high"),
    [
        (8, 10, 0.490, 0.943),  # textbook values for the Wilson 95% interval
        (0, 10, 0.000, 0.278),
        (10, 10, 0.722, 1.000),
        (45, 50, 0.786, 0.957),
    ],
)
def test_wilson_interval_matches_known_values(successes, n, low, high):
    got_low, got_high = wilson_interval(successes, n)
    assert got_low == pytest.approx(low, abs=0.002) and got_high == pytest.approx(high, abs=0.002)


def test_wilson_never_leaves_zero_to_one_and_rejects_empty_samples():
    for successes in range(0, 6):
        low, high = wilson_interval(successes, 5)
        assert 0.0 <= low <= successes / 5 <= high <= 1.0
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_the_interval_narrows_as_the_sample_grows():
    small = interval_of([1.0] * 9 + [0.0])  # 0.9 from 10 cases
    large = interval_of([1.0] * 90 + [0.0] * 10)  # 0.9 from 100 cases
    assert small.mean == large.mean == pytest.approx(0.9)
    assert (small.high - small.low) > 2 * (large.high - large.low)
    assert large.low > 0.82 and large.high < 0.96  # the point of the exercise: ~±7 points at n=100


def test_graded_metrics_use_a_normal_approximation_and_singletons_get_none():
    graded = interval_of([1.0, 0.5, 1.0, 0.0, 1.0, 0.5])
    assert graded.n == 6 and graded.low < graded.mean < graded.high
    assert graded.low >= 0.0 and graded.high <= 1.0
    assert interval_of([0.7]) is None and interval_of([]) is None


def test_metric_intervals_cover_only_metrics_with_two_or_more_values():
    results = [case("a", retrieval_hit=1.0, correctness=1.0), case("b", retrieval_hit=0.0)]
    intervals = metric_intervals(results)
    assert set(intervals) == {"retrieval_hit"} and intervals["retrieval_hit"].n == 2


# -- outcomes ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(error="boom"), ERROR),
        (
            dict(answerable=False, abstain_correct=0.0, unanswerable_abstain_rate=0.0),
            ANSWERED_UNANSWERABLE,
        ),
        (dict(answerable=False, abstain_correct=1.0, unanswerable_abstain_rate=1.0), OK),
        # declined although the passage WAS retrieved: the abstention is the problem
        (
            dict(abstain_correct=0.0, answerable_abstain_rate=1.0, retrieval_hit=1.0),
            WRONG_ABSTENTION,
        ),
        # declined because nothing useful was retrieved: retrieval is the root cause
        (dict(abstain_correct=0.0, answerable_abstain_rate=1.0, retrieval_hit=0.0), RETRIEVAL_MISS),
        (dict(correctness=0.5, retrieval_hit=1.0), WRONG_WITH_RIGHT_CONTEXT),
        (dict(correctness=0.0, retrieval_hit=0.0), RETRIEVAL_MISS),
        (dict(retrieval_hit=0.0), RETRIEVAL_MISS),  # no judge ran; retrieval alone failed
        (dict(retrieval_hit=1.0), OK),  # nothing judged, nothing wrong found
        (
            dict(correctness=1.0, retrieval_hit=0.0),
            OK,
        ),  # right answer from another passage: gold may be incomplete
        (dict(correctness=1.0, retrieval_hit=1.0, citation_valid_rate=0.5), CITATION_PROBLEM),
        (dict(correctness=1.0, retrieval_hit=1.0, citation_source_match=0.0), CITATION_PROBLEM),
        (dict(correctness=1.0, retrieval_hit=1.0, citation_valid_rate=1.0), OK),
    ],
)
def test_each_case_lands_in_one_outcome_bucket(kwargs, expected):
    assert outcome_of(case(**kwargs)) == expected
    assert case(**kwargs).outcome == expected


def test_errors_beat_everything_and_wrong_answers_beat_citation_issues():
    assert outcome_of(case(error="x", correctness=0.0, retrieval_hit=0.0)) == ERROR
    both = case(correctness=0.0, retrieval_hit=1.0, citation_valid_rate=0.0)
    assert outcome_of(both) == WRONG_WITH_RIGHT_CONTEXT


def test_the_breakdown_lists_cases_in_a_stable_order_with_ok_first():
    results = [
        case("a", retrieval_hit=0.0, correctness=0.0),
        case("b", correctness=1.0, retrieval_hit=1.0),
        case("c", retrieval_hit=0.0),
        case("d", correctness=1.0, retrieval_hit=1.0),
    ]
    assert outcome_breakdown(results) == {OK: ["b", "d"], RETRIEVAL_MISS: ["a", "c"]}


# -- question types ------------------------------------------------------------------------------


def test_metrics_are_broken_down_by_question_type():
    results = [
        case("a", tags=["lookup"], retrieval_hit=1.0, correctness=1.0),
        case("b", tags=["lookup", "table"], retrieval_hit=0.0, correctness=0.0),
        case("c", tags=["table"], retrieval_hit=1.0),
    ]
    table = by_tag(results)
    assert table["lookup"] == {"n": 2.0, "retrieval_hit": 0.5, "correctness": 0.5}
    assert table["table"] == {"n": 2.0, "retrieval_hit": 0.5, "correctness": 0.0}
    assert list(table) == ["lookup", "table"]
    assert by_tag([case("x")]) == {}


# -- the report ----------------------------------------------------------------------------------


def make_report(results):
    return EvalReport(
        name="t",
        created_at="now",
        agent="agent",
        results=results,
        summary=aggregate(results),
        counts={"cases": len(results), "answerable": len(results), "unanswerable": 0, "errors": 0},
    )


def sample_results():
    return [
        case("a", tags=["lookup"], retrieval_hit=1.0, correctness=1.0, retrieval_mrr=1.0),
        case("b", tags=["lookup"], retrieval_hit=0.0, correctness=0.0, retrieval_mrr=0.0),
        case("c", tags=["followup"], retrieval_hit=1.0, correctness=0.5, retrieval_mrr=0.5),
        case("d", tags=["followup"], retrieval_hit=1.0, correctness=1.0, retrieval_mrr=1.0),
    ]


def test_the_text_report_shows_intervals_outcomes_and_question_types():
    text = make_report(sample_results()).to_text()
    assert "95% intervals" in text and "retrieval_hit" in text and "n=4" in text
    assert "outcomes" in text and "retrieval_miss" in text and "wrong_with_right_context" in text
    assert "look at:" in text and "kbsdk ingest --report" in text  # the hint for retrieval misses
    assert "by question type" in text and "lookup" in text and "followup" in text
    assert "ok" in text.split("outcomes", 1)[1]


def test_the_markdown_report_has_the_same_sections():
    markdown = make_report(sample_results()).to_markdown()
    assert "## 95% intervals (all cases)" in markdown and "## Outcomes" in markdown
    assert "## By question type" in markdown and "| lookup | 2 |" in markdown


def test_reports_without_tags_or_with_one_case_still_render():
    text = make_report([case("only", retrieval_hit=1.0)]).to_text()
    assert "by question type" not in text and "95% intervals" not in text  # one case: no interval
    assert "outcomes" in text


def test_analysis_survives_a_save_and_load_round_trip(tmp_path):
    report = make_report(sample_results())
    report.save(tmp_path / "r.json")
    loaded = EvalReport.load(tmp_path / "r.json")
    assert loaded.outcomes() == report.outcomes()
    assert loaded.intervals() == report.intervals()
    assert loaded.results[1].outcome == RETRIEVAL_MISS
