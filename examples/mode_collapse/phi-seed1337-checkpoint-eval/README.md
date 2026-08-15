# Phi-4-mini-instruct seed-1337 checkpoint evaluation

Evaluates the base model and the seed-1337 IPS-GRPO, TTRL, and MSTTRL
checkpoints on causal, 3D, and Boolean validation sets.

Fresh evaluation seeds are `0`, `1`, `42`, and `3447`. Seed `1337` is read
from the matching original training validation event. The final output is
`result/phi_seed1337_checkpoint_eval/results.json` and contains only accuracy,
recovery rate, and mode recovery rate, plus their population summary statistics.

`run_all.sh` is resumable: an evaluation is skipped when its validation JSONL
already exists.
