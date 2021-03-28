import base64, io, torch
from alfs_char.store import ImageRepository
from alfs_char.web import app
from fastapi.testclient import TestClient
from torchvision.transforms import ToTensor
from PIL import Image as PILImage
from vnet.utils import DetectionPlot
from vnet import Boxes, Confidences, resize_boxes


client = TestClient(app)


def test_detect() -> None:
    repo = ImageRepository()
    rows = repo.filter()
    if len(rows) == 0:
        return
    image_id = "99ccedb5-449a-4092-9a40-ad119eaea964"
    row = repo.find(image_id)
    res = client.post("/detect", json=dict(data=row["data"])).json()
    boxes = Boxes(torch.tensor(res["boxes"]))
    confidences = Confidences(torch.tensor(res["confidences"]))
    to_tensor = ToTensor()
    img_tensor = to_tensor(
        PILImage.open(io.BytesIO(base64.b64decode(row["data"]))).convert("RGB")
    )
    plot = DetectionPlot(img_tensor)
    plot.draw_boxes(boxes, confidences=confidences)
    plot.save(f"store/test_detect_{image_id}.png")
