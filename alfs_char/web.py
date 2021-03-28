from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import *
import torch
import numpy as np
from PIL import Image as PILImage
import base64
from torchvision.transforms import ToPILImage, ToTensor
from vnet.transforms import inv_normalize
from vnet import Image, Boxes, Confidences, inv_scale_and_pad, shift, resize_boxes
from io import BytesIO
import base64
import asyncio
from alfs_char.config import model, model_loader, to_boxes
from alfs_char.data import test_transforms
from alfs_char import config
from logging import getLogger
from pydantic import BaseModel


logger = getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware)

model_loader.load_if_needed(model)
model.eval()
device = torch.device("cuda")


class DetectionInput(BaseModel):
    data: str  # base64 image data


class DetectionOutput(BaseModel):
    boxes: List[Tuple[float, float, float, float]]
    confidences: List[float]


@app.post("/detect")
async def detect(payload: DetectionInput) -> DetectionOutput:
    pil_img = PILImage.open(BytesIO(base64.b64decode(payload.data))).convert("RGB")
    transformed = test_transforms(image=np.array(pil_img), labels=[], bboxes=[])
    with torch.no_grad():
        image_batch = torch.stack([transformed["image"]]).to(device)
        _, _, h, w = image_batch.shape
        netout = model(image_batch)
        boxes_List, scores_List, _ = to_boxes(netout)
        boxes = boxes_List[0]
        confidences = scores_List[0]
    original_wh = (pil_img.width, pil_img.height)
    padded_wh = (w, h)
    scale, pad = inv_scale_and_pad(original_wh, padded_wh)
    boxes = shift(boxes, (-pad[0], -pad[1]))
    boxes = resize_boxes(boxes, (scale, scale))
    return DetectionOutput(
        boxes=boxes.tolist(),
        confidences=confidences.tolist(),
    )
