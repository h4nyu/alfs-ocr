import torch
import typing
from torch.utils.data import Dataset
from toolz import map
import numpy as np

import base64
from io import BytesIO
from .store import ImageRepository
from skimage.io import imread
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.ops.boxes import clip_boxes_to_image, remove_small_boxes
from PIL import Image as PILImage
import os
from object_detection.entities import Image, Boxes, Labels, resize
import cv2
import albumentations as albm
from .store import Rows
from . import config
from object_detection.transforms import RandomLayout

bbox_params = {"format": "pascal_voc", "label_fields": ["labels"]}
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
        A.OneOf(
            [
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ],
            p=0.2,
        ),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.2),
        A.Cutout(),
        A.ColorJitter(p=0.2),
        albm.RandomBrightnessContrast(),
        ToTensorV2(),
    ],
    bbox_params=bbox_params,
)


class TrainDataset(Dataset):
    def __init__(
        self,
        repo: ImageRepository,
        rows: Rows,
        mode: typing.Literal["test", "train"] = "train",
    ) -> None:
        self.repo = repo
        self.rows = rows
        self.transforms = train_transforms if mode == "train" else test_transforms

    def __getitem__(self, idx: int) -> tuple[str, Image, Boxes, Labels]:
        id = self.rows[idx]["id"]
        res = self.repo.find(id)
        image = np.array(
            PILImage.open(BytesIO(base64.b64decode(res["data"]))).convert("RGB")
        )
        h, w, _ = image.shape
        boxes = Boxes(
            clip_boxes_to_image(
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
        )
        indices = remove_small_boxes(boxes, 0.00001)
        boxes = resize(Boxes(boxes[indices]), (w, h))
        labels = Labels(torch.tensor([0 for b in boxes]))
        transed = self.transforms(image=image, bboxes=boxes, labels=labels)
        return (
            id,
            Image(transed["image"] / 255),
            Boxes(torch.tensor(transed["bboxes"])),
            Labels(torch.tensor(transed["labels"])),
        )

    def __len__(self) -> int:
        return len(self.rows)
