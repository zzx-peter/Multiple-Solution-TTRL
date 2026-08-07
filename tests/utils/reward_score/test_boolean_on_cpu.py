from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.boolean import (
    _expression_key,
    _format_expression,
    _parse_formula,
    _parse_prediction,
    compute_score,
)


GROUND_TRUTH = {
    "ground_truth_list": [
        {"formula": "x AND y"},
        {"formula": "NOT x"},
    ],
    "variables": ["x", "y"],
    "operators": ["AND", "OR", "NOT"],
    "max_depth": 2,
}


def test_boolean_score_accepts_commutative_structural_match():
    result = compute_score("Reasoning: both inputs must be 1.\nExpression: y AND x", GROUND_TRUTH)

    expected = _parse_formula("x AND y", max_depth=2)
    assert result["score"] == 1.0
    assert result["boolean_parse_success"] == 1.0
    assert result["boolean_valid"] == 1.0
    assert result["boolean_pred"] == _expression_key(expected)
    assert result["boolean_n_ground_truths"] == 2


def test_boolean_score_keeps_semantically_equal_but_structurally_distinct_modes():
    ground_truth = {**GROUND_TRUTH, "ground_truth_list": [{"formula": "x"}]}
    result = compute_score("Expression: x OR (x AND y)", ground_truth)

    assert result["boolean_parse_success"] == 1.0
    assert result["boolean_valid"] == 0.0
    assert result["score"] == 0.0
    assert result["boolean_pred"] != _expression_key(_parse_formula("x", max_depth=2))


def test_boolean_parser_rejects_out_of_space_expressions():
    assert _parse_prediction("Expression: x XOR y", max_depth=2) is None
    assert _parse_prediction("Expression: x AND True", max_depth=2) is None
    assert _parse_prediction("Expression: x AND z", max_depth=2) is None
    assert _parse_prediction("Expression: x AND (y OR NOT x)", max_depth=2) is None
    assert _parse_prediction("Expression: ((x AND x) AND x) AND x", max_depth=2) is None


def test_boolean_formatter_round_trip():
    expression = _parse_formula("NOT (x OR y) AND x", max_depth=3)
    formatted = _format_expression(expression)
    assert _parse_formula(formatted, max_depth=3) == expression


def test_default_reward_router_supports_boolean():
    result = default_compute_score(
        data_source="boolean",
        solution_str="Expression: NOT x",
        ground_truth=GROUND_TRUTH,
    )
    assert result["score"] == 1.0
    assert result["boolean_pred"]
