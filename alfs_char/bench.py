import os
from typing import List, Any
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from alfs_char.data import TrainDataset
from torchvision.transforms import ToPILImage
from io import BytesIO
import base64
from alfs_char.store import ImageRepository
import asyncio
from object_detection.models.effidet import (
    collate_fn,
)
from alfs_char.train import model, model_loader, to_boxes, criterion
from alfs_char.data import test_transforms
from alfs_char import config
from logging import getLogger

logger = getLogger(__name__)
device = torch.device("cuda")
repo = ImageRepository()
rows = repo.filter()
image_count = len(rows)
dataset = TrainDataset(repo, rows, mode="test")

@torch.no_grad()
def bench() -> None:
    loader=DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=1,
        shuffle=False,
    )
    model_loader.load_if_needed(model)
    model.eval()
    for image_batch, gt_box_batch, gt_cls_batch, ids in tqdm(loader):
        image_batch = image_batch.to(device)
        gt_box_batch = [x.to(device) for x in gt_box_batch]
        gt_cls_batch = [x.to(device) for x in gt_cls_batch]
        netout = model(image_batch)
        loss, _, _ = criterion(image_batch, netout, gt_box_batch, gt_cls_batch)
        loss = loss / len(gt_box_batch[0])
        for boxes, scores, id in zip(*to_boxes(netout), ids):
            boxes_payload = [
                dict(x0=b[0], y0=b[1], x1=b[2], y1=b[3], imageId=id, confidence=s)
                for b, s
                in zip((boxes/config.image_size).tolist(), scores.tolist())
            ]
            repo.predict(id=id, boxes=boxes_payload, loss=loss.item())




if __name__ == "__main__":
    bench()
