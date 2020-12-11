from typing import Tuple
from object_detection.entities import PyramidIdx
from object_detection.model_loader import WatchMode

confidence_threshold = 0.5
batch_size = 3
image_size = 1024

# model
channels = 128
depth = 1
lr = 1e-4

# criterion
box_weight = 4.0
topk = 11

out_dir = "/store/efficientdet"
metric: Tuple[str, WatchMode] = ("score", "max")
pretrained = True

anchor_ratios = [1.0]
anchor_scales = [1.0]
anchor_size = 2
