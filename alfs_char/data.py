import torch
import typing
from torch.utils.data import Dataset
from toolz import map
import numpy as np
import base64
from io import BytesIO
from .store import ImageRepository
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image as PILImage
import os
from object_detection.entities import (
    TrainSample,
    ImageId,
    Image,
    YoloBoxes,
    Labels,
)
import cv2
import albumentations as albm
from .store import Rows
from . import config
from .transforms import RandomLayout

bbox_params = {"format": "yolo", "label_fields": ["labels"]}
test_transforms = albm.Compose(
    [
        albm.LongestMaxSize(max_size=config.image_size),
        albm.PadIfNeeded(
            min_width=config.image_size,
            min_height=config.image_size,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)

train_transforms = albm.Compose(
    [
        albm.LongestMaxSize(max_size=config.image_size),
        albm.PadIfNeeded(
            min_width=config.image_size,
            min_height=config.image_size,
            border_mode=cv2.BORDER_CONSTANT,
            ),
        RandomLayout(config.image_size, config.image_size, (0.8, 2.0)),
        albm.RandomBrightnessContrast(),
        ToTensorV2(),
        ],
    bbox_params=bbox_params,
)


class TrainDataset(Dataset):
    def __init__(self,
            repo: ImageRepository,
            rows: Rows,
            mode: typing.Literal["test", "train"]="train",
        ) -> None:
        self.repo = repo
        self.rows = rows
        self.transforms = train_transforms if mode == "train" else test_transforms

    def __getitem__(self, idx: int) -> TrainSample:
        id = self.rows[idx]["id"]
        res = self.repo.find(id)
        image = np.array(PILImage.open(BytesIO(base64.b64decode(res["data"]))).convert('RGB'))
        boxes = YoloBoxes(
            torch.tensor(
                [
                    [
                        (b["x0"] + b["x1"]) / 2,
                        (b["y1"] + b["y0"]) / 2,
                        b["x1"] - b["x0"],
                        b["y1"] - b["y0"],
                    ]
                    for b in res["boxes"]
                ]
            ).clamp(max=1.0 - 1e-3, min=0.0 + 1e-3)
        )
        labels = Labels(torch.tensor([0 for b in boxes]))
        res = self.transforms(image=image, bboxes=boxes, labels=labels)
        return (
            ImageId(id),
            Image(res['image'] / 255),
            YoloBoxes(torch.tensor(res['bboxes'])),
            Labels(torch.tensor(res['labels'])),
        )

    def __len__(self) -> int:
        return len(self.rows)
