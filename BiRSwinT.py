# --------------------------------------------------------
# Copyright (c) 2022 BiRSwinT.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
from models import build_model
from context import ctx


class BiRSwinT(torch.nn.Module):
    """BiRSwinT for Driver Behavior Recognition.
    The network accepts a 3 * 224 * 224 input, and the output of each brranch has shape 49*768.
    Attributes:
        features, torch.nn.Module: Swin-Transformer layers and shortcuts.
        fc, torch.nn.Module: 10.
    """

    def __init__(self, config):
        """Declare all needed layers."""
        super().__init__()

        round = ctx.current_round
        adjustable = config.ROUNDS[round]["ADJUSTABLE"]

        self.global_representation_branch = build_model(
            config,
            post_process=False,
            shortcut=False,
            adjustable=adjustable,
        )
        self.local_residual_branch = build_model(config, post_process=False, shortcut=True, adjustable=adjustable)
        if round == 0:
            checkpoint1 = torch.load(config.DATA.SWIN_TRANSFORMER_CHECKPOINT_PATH, map_location="cpu")
            checkpoint2 = torch.load("swin_small_patch4_window7_224.pth", map_location="cpu")
            part_sd = {k: v for k, v in checkpoint2.items() if k not in ["head.weight", "head.bias"]}
            self.global_representation_branch.load_state_dict(checkpoint1["model"], strict=False)
            self.local_residual_branch.state_dict().update(part_sd)
        else:
            checkpoint = torch.load(ctx.latest_round_result, map_location="cpu")
            self.load_state_dict(checkpoint["state_dict"], strict=False)

        # Linear classifier.
        self.fc = torch.nn.Linear(768 * 768, config.MODEL.NUM_CLASSES, bias=True)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape.
        Returns:
            Score, torch.autograd.Variable of shape N*10.
        """
        N = X.size()[0]
        X1 = self.global_representation_branch(X)
        X2 = self.local_residual_branch(X)

        X = torch.bmm(torch.transpose(X1, 1, 2), X2) / (768)  # Bilinear
        X = torch.reshape(X, (N, 768 * 768))
        X = torch.sqrt(torch.abs(X) + 1e-10) * torch.sign(X)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        return X
