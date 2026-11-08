from __future__ import annotations

from eval.metrics import count_leaves, score_prediction


def test_exact_match_scores_one() -> None:
    truth = {"a": 1, "b": "hello", "nested": {"c": True}}
    assert score_prediction(truth, truth).score == 1.0


def test_missing_and_extra_fields_are_penalised() -> None:
    truth = {"a": 1, "b": 2}
    assert score_prediction({"a": 1}, truth).score == 0.5
    assert score_prediction({"a": 1, "b": 2, "c": 3}, truth).score < 1.0


def test_string_similarity_is_partial() -> None:
    score = score_prediction({"name": "Atlas Robotics"}, {"name": "Atlas Robotics GmbH"})
    assert 0.5 < score.score < 1.0


def test_number_tolerance() -> None:
    assert score_prediction({"total": 5533.5001}, {"total": 5533.5}).score == 1.0
    assert score_prediction({"total": 5000}, {"total": 5533.5}).score == 0.0


def test_arrays_are_index_aligned() -> None:
    truth = {"items": [{"name": "a"}, {"name": "b"}]}
    partial = {"items": [{"name": "a"}]}
    assert score_prediction(partial, truth).score == 0.5
    extra = {"items": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
    assert score_prediction(extra, truth).score < 1.0


def test_per_field_breakdown() -> None:
    score = score_prediction({"total": 10, "name": "x"}, {"total": 10, "name": "y"})
    assert score.per_field["total"] == 1.0
    assert score.per_field["name"] < 1.0


def test_count_leaves() -> None:
    assert count_leaves({"a": 1, "b": {"c": 2}}) == 2
    assert count_leaves([{"a": 1}, {"a": 2}]) == 2
    assert count_leaves([]) == 1
