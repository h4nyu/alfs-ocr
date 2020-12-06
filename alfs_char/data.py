import torch
import typing
from torch.utils.data import Dataset
from urllib.parse import urljoin
from toolz import map
import base64
from io import BytesIO
from torchvision import transforms
from PIL import Image as PILImage
import requests
from object_detection.entities import (
    TrainSample,
    ImageId,
    Image,
    YoloBoxes,
    Labels,
)


class TrainDataset(Dataset):
    def __init__(
        self,
        url: str,
    ) -> None:
        ...
        self.url = url
        self.rows: typing.List[typing.Any] = []

        res = requests.post(
            urljoin(url, "/api/v1/char-image/filter"), json={"hasBox": True}
        )
        if res.status_code == 200:
            self.rows = res.json()
        self.cache: typing.Dict[str, typing.Any] = {}

    def __getitem__(self, idx: int) -> TrainSample:
        id = self.rows[idx]['id']
        if id in self.cache:
            return self.cache[id]
        res = requests.post(
            urljoin(self.url, "/api/v1/char-image/find"), json={"id": id}
        ).json()

        img = Image(
            transforms.ToTensor()(PILImage.open(BytesIO(base64.b64decode(res["data"]))))
        )
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
            )
        )
        labels = Labels(torch.tensor([0 for b in boxes]))
        self.cache[id] = (ImageId(id), img, boxes, labels)
        return self.cache[id]

    def __len__(self) -> int:
        return len(self.rows)
        ...
