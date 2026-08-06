from verl.utils.reward_score.three_d import (
    _format_structure,
    _parse_prediction,
    _structure_key,
    compute_score,
)


GROUND_TRUTH = {
    "ground_truth_list": [
        {
            "grid_size": 3,
            "layers": [
                [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
        },
        {
            "grid_size": 3,
            "layers": [
                [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
        },
    ],
    "grid_size": 3,
    "max_height": 3,
    "max_blocks": 2,
}


def test_three_d_score_accepts_a_canonical_ground_truth():
    response = """Reasoning:
The occupied positions can be supported in two layers.
BEGIN_STRUCTURE
Structure:
Layer 1:
1 0 0
0 1 0
0 0 0
Layer 2:
1 0 0
0 0 0
0 0 0
END_STRUCTURE
"""
    result = compute_score(response, GROUND_TRUTH)

    assert result["score"] == 1.0
    assert result["3d_parse_success"] == 1.0
    assert result["3d_valid"] == 1.0
    assert result["3d_n_ground_truths"] == 2
    assert result["3d_pred"] == _structure_key(_parse_prediction(response))


def test_three_d_parser_rejects_floating_blocks_and_bad_layer_numbers():
    floating = """Structure:
Layer 1:
0 0 0
0 1 0
0 0 0
Layer 2:
1 0 0
0 0 0
0 0 0
"""
    skipped_layer = """Structure:
Layer 2:
1 0 0
0 0 0
0 0 0
"""

    assert _parse_prediction(floating, max_height=3) is None
    assert _parse_prediction(skipped_layer, max_height=3) is None
    assert compute_score(floating, GROUND_TRUTH)["3d_parse_success"] == 0.0


def test_three_d_formatter_round_trip():
    structure = (
        ((1, 0, 0), (0, 0, 0), (0, 0, 0)),
        ((1, 0, 0), (0, 0, 0), (0, 0, 0)),
    )
    assert _parse_prediction(_format_structure(structure), max_height=3) == structure
