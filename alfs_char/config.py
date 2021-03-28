import torch
import torch_optimizer as optim
from typing import Tuple, List, Any
from vnet.model_loader import WatchMode
from vnet.effidet import (
    EfficientDet,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
)
from vnet.model_loader import (
    ModelLoader,
    BestWatcher,
)
from vnet.backbones.effnet import (
    EfficientNetBackbone,
)
from vnet.metrics import MeanAveragePrecision

confidence_threshold = 0.1
iou_threshold = 0.2
batch_size = 5
image_size = 768

backbone_idx = 4
# model
channels = 64
depth = 1
lr = 1e-4
out_ids: List[Any] = [5, 6, 7]
num_classes = 1

# criterion
box_weight = 1.0
cls_weight = 1.5

metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

anchor_ratios = [1.0]
anchor_scales = [1.0]
anchor_size = 1

anchors = Anchors(
    size=anchor_size,
    ratios=anchor_ratios,
    scales=anchor_scales,
)

out_dir = f"/store/efficientdet-{anchors.num_anchors}-{''.join([str(i) for i in out_ids])}-{image_size}"

device = torch.device("cuda")
use_amp = True

Metrics = lambda: MeanAveragePrecision(
    num_classes=num_classes,
    iou_threshold=0.5,
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

topk = anchors.num_anchors * len(out_ids) * 10

criterion = Criterion(
    topk=topk,
    box_weight=box_weight,
    cls_weight=cls_weight,
)

to_boxes = ToBoxes(
    confidence_threshold=confidence_threshold,
    iou_threshold=iou_threshold,
)

optimizer = optim.RAdam(
    model.parameters(),
    lr=lr,
    eps=1e-8,
    betas=(0.9, 0.999),
)
