# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


@register("ttrl")
class TTRLRewardManager(NaiveRewardManager):
    """Naive rule-based reward manager used after TTRL rewrites ground truth."""

    pass
