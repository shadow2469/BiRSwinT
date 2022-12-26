# --------------------------------------------------------
# Copyright (c) 2022 BiRSwinT Authors.
# Licensed under The MIT License.
# --------------------------------------------------------

import torch


def build_scheduler(config, optimizer, train_data_size):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, int(train_data_size / config.DATA.BATCH_SIZE), T_mult=2, eta_min=0, last_epoch=-1, verbose=False
    )
