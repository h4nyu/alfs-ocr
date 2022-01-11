import os
from typing import Any
from omegaconf import OmegaConf
from vision_tools.backbone import EfficientNet
from vision_tools.neck import CSPPAFPN
from vision_tools.yolox import YOLOX
from vision_tools.utils import Checkpoint
from datetime import datetime


def get_model_name(cfg: Any) -> str:
    return f"{cfg.name}-{cfg.feat_range[0]}-{cfg.feat_range[1]}-{cfg.hidden_channels}-{cfg.backbone.name}"


def get_model(cfg: Any) -> YOLOX:
    backbone = EfficientNet(name=cfg.backbone.name)
    neck = CSPPAFPN(
        in_channels=backbone.channels[cfg.feat_range[0] : cfg.feat_range[1]],
        strides=backbone.strides[cfg.feat_range[0] : cfg.feat_range[1]],
    )
    model = YOLOX(
        backbone=backbone,
        neck=neck,
        hidden_channels=cfg.hidden_channels,
        num_classes=cfg.num_classes,
        feat_range=cfg.feat_range,
        box_iou_threshold=cfg.box_iou_threshold,
        score_threshold=cfg.score_threshold,
    )
    return model


def get_checkpoint(cfg: Any) -> Checkpoint:
    return Checkpoint[YOLOX](
        root_path=os.path.join(cfg.root_dir, get_model_name(cfg)),
        default_score=0.0,
    )
