# --------------------------------------------------------
# Copyright (c) 2022 BiRSwinT Authors.
# Licensed under The MIT License. 
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP
from torchvision import models


def build_model(config, post_process=True, shortcut=True, adjustable=True):
    model_type = config.MODEL.TYPE
    if model_type == "swin":
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            shortcut=shortcut,
            post_process=post_process,
            adjustable=adjustable
        )
    elif model_type == "swin_mlp":
        model = SwinMLP(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
            in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
            depths=config.MODEL.SWIN_MLP.DEPTHS,
            num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
            window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN_MLP.APE,
            patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
    elif model_type == "resnet18":
        model = models.resnet18(pretrained=True)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
