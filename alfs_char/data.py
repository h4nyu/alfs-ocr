import torch
from typing import *
from typing_extensions import Literal
from torch.utils.data import Dataset
from toolz import map
import numpy as np

import base64
from io import BytesIO
from .store import ImageRepository
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.ops.boxes import clip_boxes_to_image, remove_small_boxes
from PIL import Image as PILImage
import os
import albumentations as albm
from .store import Rows

bbox_params = {"format": "pascal_voc", "label_fields": ["labels"]}

Transform = lambda image_size: A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)
