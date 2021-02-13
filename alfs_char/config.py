import torch
from typing import Tuple, List, Any
from object_detection.model_loader import WatchMode
from object_detection.models.effidet import (
    EfficientDet,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
)
from object_detection.model_loader import (
    ModelLoader,
    BestWatcher,
)
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)
from object_detection.metrics import MeanAveragePrecision

confidence_threshold = 0.35
iou_threshold = 0.7
batch_size = 2
image_size = 1024

backbone_idx = 4
# model
channels = 64
depth = 1
lr = 1e-4
out_ids: List[Any] = [4, 5, 6]
num_classes = 1

# criterion
box_weight = 100.0
topk = 29

out_dir = "/store/efficientdet-456"
metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

anchor_ratios = [1.0]
anchor_scales = [1.0, 1.25, 1.5]
anchor_size = 2

device = torch.device("cuda")
use_amp = True
metrics = MeanAveragePrecision(
    num_classes=num_classes,
    iou_threshold=0.5,
)


anchors = Anchors(
    size=anchor_size,
    ratios=anchor_ratios,
    scales=anchor_scales,
)
backbone = EfficientNetBackbone(backbone_idx, out_channels=channels, pretrained=True)
model = EfficientDet(
    num_classes=num_classes,
    channels=channels,
    backbone=backbone,
    anchors=anchors,
    out_ids=out_ids,
    box_depth=depth,
).to(device)

model_loader = ModelLoader(
    out_dir=out_dir,
    key=metric[0],
    best_watcher=BestWatcher(mode=metric[1]),
)
criterion = Criterion(
    topk=topk,
    box_weight=box_weight,
)
