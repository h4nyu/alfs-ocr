from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Any
import torch
import numpy as np
from uvicorn import Config, Server
from PIL import Image as PILImage
import base64
from torchvision.transforms import ToPILImage
from io import BytesIO
import base64
import asyncio
from starlette.middleware.cors import CORSMiddleware
from alfs_char.train import model, model_loader, to_boxes
from alfs_char.data import test_transforms
from alfs_char import config

app = FastAPI()


class UploadPayload(BaseModel):
    data: str

model_loader.load_if_needed(model)
device = torch.device("cuda")

@torch.no_grad()
def detect(payload: UploadPayload) -> Any:
    image = np.array(
        PILImage.open(BytesIO(base64.b64decode(payload.data))).convert("RGB")
    )
    img_tensor = test_transforms(image=image, labels=[], bboxes=[])["image"]
    batch = torch.stack([img_tensor / 255]).to(device)
    netout = model(batch)
    boxes_list, scores_list = to_boxes(netout)
    out_boxes = boxes_list[0] / config.image_size
    out_scores = scores_list[0]
    out_img = ToPILImage()(batch[0].cpu())
    buffer = BytesIO()
    out_img.save(buffer, format="JPEG")
    return {
        "image": base64.b64encode(buffer.getvalue()),
        "boxes": out_boxes.tolist(),
        "scores": out_scores.tolist(),
    }


app.post("/api/upload-image")(detect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    server = Server(
        config=Config(
            app=app,
            host="0.0.0.0",
            port=5000,
            loop=loop,
            log_level="info",
        )
    )
    loop.run_until_complete(server.serve())
