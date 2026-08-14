# Qwen3-1.7B seed-1337 checkpoint evaluation

This directory is intentionally separate from the training launch scripts. It
evaluates the base model and the IPS-GRPO, TTRL, and MSTTRL checkpoints trained
with seed 1337 on evaluation seeds `0`, `1`, `42`, `1337`, and `3447`.

The fresh runs use the original task/method training script with Hydra overrides
for `trainer.val_only=True` and, for trained models,
`trainer.resume_mode=resume_path`. Seed 1337 is taken from the matching original
training validation event as requested and is not rerun.

`run_all.sh` runs only the four fresh seeds. It writes trajectories and logs to
`result/qwen_seed1337_checkpoint_eval/`, then creates `results.json` containing
all five seeds plus the mean and population variance of `acc`, `recovery_rate`,
and `mode_recovery_rate`.
