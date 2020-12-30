import os
from typing import List, Any
import torch
from torch.utils.data import DataLoader
from alfs_char.data import PredictDataset
from torchvision.transforms import ToPILImage
from io import BytesIO
import base64
from alfs_char.store import ImageRepository
import asyncio
from object_detection.models.effidet import (
    Predictor,
    prediction_collate_fn,
)
from alfs_char.train import model, model_loader, to_boxes
from alfs_char.data import test_transforms
from alfs_char import config
from logging import getLogger

logger = getLogger(__name__)
device = torch.device("cuda")
repo = ImageRepository()
rows = repo.filter()
image_count = len(rows)
dataset = PredictDataset(repo, rows)

@torch.no_grad()
def bench() -> None:
    loader=DataLoader(
        dataset,
        collate_fn=prediction_collate_fn,
        batch_size=config.batch_size,
        shuffle=True,
    )
    model_loader.load_if_needed(model)
    model.eval()
    for images, ids in loader:
        images = images.to(device)
        netout = model(images)
        for boxes, scores, id in zip(*to_boxes(netout), ids):
            boxes_payload = [
                dict(x0=b[0], y0=b[1], x1=b[2], y1=b[3], imageId=id)
                for b
                in (boxes/config.image_size).tolist()
            ]
            repo.predict(id=id, boxes=boxes_payload)




if __name__ == "__main__":
    bench()
