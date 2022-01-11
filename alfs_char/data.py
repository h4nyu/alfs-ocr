import torch
from torch import Tensor
from typing import *
from torch.utils.data import Dataset
from toolz import map
import numpy as np

import base64
from io import BytesIO
import cv2
from .store import ImageRepository
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.ops.boxes import clip_boxes_to_image, remove_small_boxes
from PIL import Image as PILImage
import os
import albumentations as albm
from .store import Rows
from vision_tools.interface import TrainBatch, TrainSample
from vision_tools.box import resize_boxes

bbox_params = {"format": "pascal_voc", "label_fields": ["labels"]}

Transform = lambda image_size: A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)

TrainTransform = lambda image_size: A.Compose(
    [
        A.PadIfNeeded(
            min_width=image_size,
            min_height=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.Rotate(limit=(-5, 5), p=1.0, border_mode=0),
        A.OneOf(
            [
                A.Blur(blur_limit=7, p=0.5),
                A.MotionBlur(blur_limit=7, p=0.5),
                A.MotionBlur(blur_limit=13, p=0.5),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ],
            p=0.2,
        ),
        A.Cutout(),
        A.ColorJitter(p=0.2),
        albm.RandomBrightnessContrast(),
        A.HueSaturationValue(
            p=0.3,
            hue_shift_limit=15,
            sat_shift_limit=20,
            val_shift_limit=15,
        ),
        A.OneOf(
            [
                A.RandomSizedBBoxSafeCrop(
                    height=image_size,
                    width=image_size,
                    p=1.0,
                ),
                A.RandomResizedCrop(height=image_size, width=image_size),
            ],
            p=1.0,
        ),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)


class TrainDataset(Dataset):
    def __init__(
        self,
        repo: ImageRepository,
        rows: Rows,
        transform: Any,
    ) -> None:
        self.repo = repo
        self.rows = rows
        self.transform = transform

    def __getitem__(self, idx: int) -> TrainSample:
        id = self.rows[idx]["id"]
        res = self.repo.find(id)
        image = (
            np.array(
                PILImage.open(BytesIO(base64.b64decode(res["data"]))).convert("RGB")
            )
        )
        h, w, _ = image.shape
        boxes = clip_boxes_to_image(
            torch.tensor(
                [
                    [
                        b["x0"],
                        b["y0"],
                        b["x1"],
                        b["y1"],
                    ]
                    for b in res["boxes"]
                ]
            ),
            (1.0, 1.0),
        )
        indices = remove_small_boxes(boxes, 0.00001)
        boxes = resize_boxes(boxes[indices], (w, h))
        labels = torch.tensor([0 for b in boxes])
        transed = self.transform(image=image, bboxes=boxes, labels=labels)
        return dict(
            id=id,
            image=transed["image"] / 255,
            boxes=torch.tensor(transed["bboxes"]),
            labels=torch.tensor(transed["labels"]),
        )

    def __len__(self) -> int:
        return len(self.rows)


def collate_fn(
    batch: list[TrainSample],
) -> TrainBatch:
    images: list[Tensor] = []
    box_batch: list[Tensor] = []
    label_batch: list[Tensor] = []
    for row in batch:
        images.append(row["image"])
        box_batch.append(row["boxes"])
        label_batch.append(row["labels"])
    return TrainBatch(
        image_batch=torch.stack(images),
        box_batch=box_batch,
        label_batch=label_batch,
    )
