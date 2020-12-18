from typing import Tuple, List, Any
from object_detection.entities import PyramidIdx
from object_detection.model_loader import WatchMode

confidence_threshold = 0.5
batch_size = 2
image_size = 1024

backbone_idx = 2
# model
channels = 128
depth = 1
lr = 1e-4
out_ids: List[Any] = [4, 5, 6, 7]

# criterion
box_weight = 5.0
topk = 25

out_dir = "/store/efficientdet"
metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

anchor_ratios = [1.0]
anchor_scales = [1.0]
anchor_size = 3
