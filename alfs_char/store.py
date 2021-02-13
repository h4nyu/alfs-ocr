import requests
from urllib.parse import urljoin
import os
import typing

Box = typing.TypedDict(
    "Box",
    {
        "x0": float,
        "y0": float,
        "x1": float,
        "y1": float,
        "confidence": typing.Optional[float],
    },
)
Row = typing.TypedDict(
    "Row",
    {
        "id": str,
        "data": str,
        "hasBox": typing.Optional[bool],
        "boxes": typing.List[Box],
    },
)
Rows = typing.List[Row]


class ImageRepository:
    def __init__(self, url: str = os.getenv("STORE_URL", "")) -> None:
        self.url = url
        self.cache: typing.Dict[str, Row] = {}

    def filter(self) -> Rows:
        rows = requests.post(
            urljoin(self.url, "/api/v1/image/filter"),
            json={"state": "Done"},
        ).json()
        return [r for r in rows if r["boxCount"] > 0]

    def find(self, id: str) -> Row:
        if id not in self.cache:
            img = requests.post(
                urljoin(self.url, "/api/v1/image/find"), json={"id": id}
            ).json()
            boxes = requests.post(
                urljoin(self.url, "/api/v1/box/filter"),
                json={"imageId": id, "isGrandTruth": True},
            ).json()
            img["boxes"] = boxes
            self.cache[id] = img
        return self.cache[id]

    def predict(
        self, id: str, boxes: typing.List[Box], loss: typing.Optional[float] = None
    ) -> None:
        res = requests.post(
            urljoin(self.url, "/api/v1/box/predict"),
            json={"imageId": id, "boxes": boxes, "loss": loss},
        )
        res.raise_for_status()
