from verl.trainer.ppo.ttrl_utils import _majority_vote


def test_ttrl_majority_vote_routes_three_d_and_preserves_structure_metadata():
    ground_truth = {
        "ground_truth_list": [],
        "grid_size": 3,
        "max_height": 3,
        "max_blocks": 2,
    }
    first = """Structure:
Layer 1:
1 0 0
0 0 0
0 0 0
"""
    second = """Structure:
Layer 1:
0 1 0
0 0 0
0 0 0
"""

    pseudo_gt, majority_keys, majority_ratio = _majority_vote(
        "3d",
        [first, first, second],
        ground_truth,
    )

    assert majority_ratio == 2 / 3
    assert majority_keys == ["[[[1,0,0],[0,0,0],[0,0,0]]]"]
    assert pseudo_gt["grid_size"] == 3
    assert pseudo_gt["max_height"] == 3
    assert pseudo_gt["ground_truth_list"] == [
        {
            "grid_size": 3,
            "layers": [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]],
        }
    ]
