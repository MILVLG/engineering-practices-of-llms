# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import modellink
from modellink.tasks.trainer.launcher import AutoTrainer


def launch():
    trainer = AutoTrainer()
    trainer.train()


if __name__ == '__main__':
    launch()
