# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from modellink.tasks.trainer.base import BaseTrainer
from modellink.tasks.rl.dpo import DPOTrainer


class AutoTrainer:
    """
    A trainer is used to automatically determine which scenario (common pt, sft, rm, dpo) to use.
    """

    def __init__(self):
        initialize_megatron()

        self.args = get_args()

        if self.args.stage == "dpo":
            self.trainer = DPOTrainer()
        else:
            self.trainer = BaseTrainer()

    def train(self):
        self.trainer.train()
