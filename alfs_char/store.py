import requests
from urllib.parse import urljoin
import os
from typing import *
from typing_extensions import TypedDict
from toolz import unique, valfilter, groupby

Box = TypedDict(
    "Box",
    {
        "x0": float,
        "y0": float,
        "x1": float,
        "y1": float,
        "confidence": Optional[float],
    },
)
Row = TypedDict(
    "Row",
    {
        "id": str,
        "data": str,
        "hasBox": Optional[bool],
        "boxes": List[Box],
    },
)
Rows = List[Row]


class ImageRepository:
    def __init__(self, url: str = os.getenv("STORE_URL", "")) -> None:
        self.url = url
        self.cache: Dict[str, Row] = {}

    def filter(self) -> Rows:
        rows = requests.post(
            urljoin(self.url, "/api/v1/box/filter"),
            json={},
        ).json()
        g = groupby(lambda x: x["imageId"], rows)
        g = valfilter(lambda x: len(x) > 2, g)
        image_ids = list(g.keys())

        rows = requests.post(
            urljoin(self.url, "/api/v1/image/filter"),
            json=dict(ids=image_ids),
        ).json()
        return rows

    def find(self, id: str) -> Row:
        if id not in self.cache:
            img = requests.post(
                urljoin(self.url, "/api/v1/image/find"),
                json={"id": id, "hasData": True},
            ).json()
            boxes = requests.post(
                urljoin(self.url, "/api/v1/box/filter"),
                json={"imageId": id, "isGrandTruth": True},
            ).json()
            img["boxes"] = boxes
            self.cache[id] = img
        return self.cache[id]

    def predict(self, id: str, boxes: List[Box], loss: Optional[float] = None) -> None:
        res = requests.post(
            urljoin(self.url, "/api/v1/box/predict"),
            json={"imageId": id, "boxes": boxes, "loss": loss},
        )
        res.raise_for_status()
