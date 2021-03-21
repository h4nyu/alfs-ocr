import requests
from urllib.parse import urljoin
import os
from typing import *
from typing_extensions import TypedDict

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
            urljoin(self.url, "/api/v1/image/filter"),
            json={"state": "Done"},
        ).json()
        return [r for r in rows if r["boxCount"] > 0]

    def find(self, id: str) -> Row:
        if id not in self.cache:
            img = requests.post(
                    urljoin(self.url, "/api/v1/image/find"), json={"id": id, "hasData": True}
            ).json()
            boxes = requests.post(
                urljoin(self.url, "/api/v1/box/filter"),
                json={"imageId": id, "isGrandTruth": True},
            ).json()
            img["boxes"] = boxes
            self.cache[id] = img
        return self.cache[id]

    def predict(
        self, id: str, boxes: List[Box], loss: Optional[float] = None
    ) -> None:
        res = requests.post(
            urljoin(self.url, "/api/v1/box/predict"),
            json={"imageId": id, "boxes": boxes, "loss": loss},
        )
        res.raise_for_status()
