import requests
from urllib.parse import urljoin
import os
import typing

Box = typing.TypedDict("Box", {"x0": float, "y0": float, "x1": float, "y1": float})
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
        self.cache:typing.Dict[str,Row] = {}

    def filter(self) -> Rows:
        return requests.post(
                urljoin(self.url, "/api/v1/char-image/filter"), json={"hasBox": True, "state": "Done"}
        ).json()

    def find(self, id: str) -> Row:
        if(id not in self.cache):
            self.cache[id] = requests.post(
                urljoin(self.url, "/api/v1/char-image/find"), json={"id": id}
            ).json()
        return self.cache[id]
