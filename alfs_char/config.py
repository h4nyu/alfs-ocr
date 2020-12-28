from typing import Tuple, List, Any
from object_detection.entities import PyramidIdx
from object_detection.model_loader import WatchMode

confidence_threshold = 0.3
iou_threshold = 0.7
batch_size = 2
image_size = 1024

backbone_idx = 4
# model
channels = 64
depth = 1
lr = 1e-4
out_ids: List[Any] = [5, 6]

# criterion
box_weight = 60.0
topk = 29

out_dir = "/store/efficientdet"
metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

anchor_ratios = [1.0]
anchor_scales = [1.0, 1.25, 1.5]
anchor_size = 2