# --------------------------------------------------------
# Copyright (c) 2022 BiRSwinT
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from torch import optim as optim
from context import ctx


def build_optimizer(config, model):
    return optim.Adam(model.parameters(), lr=config.ROUNDS[ctx.current_round]["LEARNING_RATE"])
