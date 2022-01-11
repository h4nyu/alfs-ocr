from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import *
import torch
import numpy as np
from PIL import Image as PILImage
import base64
from torchvision.transforms import ToPILImage, ToTensor
from vision_tools.box import shift
from io import BytesIO
import base64
import asyncio
from alfs_char.data import Transform
from alfs_char.yolox import get_checkpoint, get_model
from logging import getLogger
from omegaconf import OmegaConf
from pydantic import BaseModel


logger = getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware)

cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config/yolox.yaml"))
model = get_model(cfg)
model.eval()
checkpoint = get_checkpoint(cfg)
checkpoint.load_if_exists(model=model, device=cfg.device)
transform = Transform(cfg.image_size)


class DetectionInput(BaseModel):
    data: str  # base64 image data


class DetectionOutput(BaseModel):
    boxes: List[Tuple[float, float, float, float]]


def inv_scale_and_pad(
    original: Tuple[int, int], padded: Tuple[int, int]
) -> Tuple[float, Tuple[float, float]]:
    original_w, original_h = original
    padded_w, padded_h = padded
    original_longest = max(original)
    if original_longest == original_w:
        scale = original_longest / padded_w
        pad = (padded_h - original_h / scale) / 2
        return scale, (0, pad)
    else:
        scale = original_longest / padded_h
        pad = (padded_w - original_w / scale) / 2
        return scale, (pad, 0)

@app.post("/detect")
async def detect(payload: DetectionInput) -> DetectionOutput:
    pil_img = PILImage.open(BytesIO(base64.b64decode(payload.data))).convert("RGB")
    transformed = transform(image=np.array(pil_img), labels=[], bboxes=[])
    with torch.no_grad():
        image_batch = torch.stack([transformed["image"] / 255] ).to(cfg.device)
        _, _, h, w = image_batch.shape
        netout = model(image_batch)
        boxes = netout["box_batch"][0]
    original_wh = (pil_img.width, pil_img.height)
    padded_wh = (w, h)
    # print(boxes)
    scale, pad = inv_scale_and_pad(original_wh, padded_wh)
    boxes = shift(boxes, (-pad[0], -pad[1]))
    boxes = boxes * scale
    return DetectionOutput(
        boxes=boxes.tolist(),
    )
