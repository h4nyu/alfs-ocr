from torch.cuda.amp import GradScaler, autocast
import torch
from tqdm import tqdm
from typing import Any
from torch import Tensor
from torch.utils.data import DataLoader
from vnet.transforms import inv_normalize
from vnet.meters import MeanMeter
from vnet import (
    Boxes,
    ImageBatch,
    Labels,
)
from vnet.effidet import (
    EfficientDet,
    Criterion,
    Visualize,
    Anchors,
)
from alfs_char.data import TrainDataset
from vnet.metrics import MeanPrecition
from alfs_char import config
from alfs_char.config import model, model_loader, criterion, optimizer
from alfs_char.store import ImageRepository
from logging import (
    getLogger,
)

logger = getLogger(config.out_dir)

visualize = Visualize(config.out_dir, "test", limit=6, box_limit=100)


def collate_fn(
    batch: list[Any],
) -> tuple[ImageBatch, list[Boxes], list[Labels], list[str]]:
    images: list[Any] = []
    id_batch: list[str] = []
    box_batch: list[Boxes] = []
    label_batch: list[Labels] = []
    for id, img, boxes, labels in batch:
        c, h, w = img.shape
        images.append(img)
        box_batch.append(boxes)
        id_batch.append(id)
        label_batch.append(labels)
    return (
        ImageBatch(torch.stack(images)),
        box_batch,
        label_batch,
        id_batch,
    )


def train(epochs: int) -> None:
    device = config.device
    repo = ImageRepository()
    rows = repo.filter()
    image_count = len(rows)
    train_dataset = TrainDataset(
        repo,
        rows[: int(image_count * 0.8)],
    )
    test_dataset = TrainDataset(
        repo,
        rows[int(image_count * 0.8) :],
        mode="test",
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size * 3,
        num_workers=config.batch_size * 3,
        shuffle=True,
    )
    to_boxes = config.to_boxes
    scaler = GradScaler()
    logs: dict[str, float] = {}

    def train_step() -> None:
        loss_meter = MeanMeter()
        box_loss_meter = MeanMeter()
        label_loss_meter = MeanMeter()
        for i, (
            image_batch,
            gt_box_batch,
            gt_label_batch,
            _,
        ) in tqdm(enumerate(train_loader)):
            if i % 50 == 0:
                eval_step()
                log()
            model.train()
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            optimizer.zero_grad()
            with autocast(enabled=config.use_amp):
                netout = model(image_batch)
                loss, box_loss, label_loss = criterion(
                    image_batch,
                    netout,
                    gt_box_batch,
                    gt_label_batch,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item())
            box_loss_meter.update(box_loss.item())
            label_loss_meter.update(label_loss.item())

            logs["train_loss"] = loss_meter.get_value()
            logs["train_box"] = box_loss_meter.get_value()
            logs["train_label"] = label_loss_meter.get_value()

    @torch.no_grad()
    def eval_step() -> None:
        model.eval()
        loss_meter = MeanMeter()
        box_loss_meter = MeanMeter()
        label_loss_meter = MeanMeter()
        metrics = config.Metrics()
        for image_batch, gt_box_batch, gt_label_batch, ids in tqdm(test_loader):
            image_batch = image_batch.to(device)
            gt_box_batch = [x.to(device) for x in gt_box_batch]
            gt_label_batch = [x.to(device) for x in gt_label_batch]
            netout = model(image_batch)
            loss, box_loss, label_loss = criterion(
                image_batch, netout, gt_box_batch, gt_label_batch
            )
            box_batch, confidence_batch, label_batch = to_boxes(netout)

            loss_meter.update(loss.item())
            box_loss_meter.update(box_loss.item())
            label_loss_meter.update(label_loss.item())

            for boxes, gt_boxes, labels, gt_labels, confidences in zip(
                box_batch, gt_box_batch, label_batch, gt_label_batch, confidence_batch
            ):
                metrics.add(
                    boxes=boxes,
                    confidences=confidences,
                    labels=labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                )

        logs["test_loss"] = loss_meter.get_value()
        logs["test_box"] = box_loss_meter.get_value()
        logs["test_label"] = label_loss_meter.get_value()
        logs["score"], _ = metrics()
        visualize(
            image_batch,
            (box_batch, confidence_batch, label_batch),
            (gt_box_batch, gt_label_batch),
        )
        model_loader.save_if_needed(
            model,
            logs[model_loader.key],
        )

    def log() -> None:
        logger.info(",".join([f"{k}={v:.3f}" for k, v in logs.items()]))

    model_loader.load_if_needed(model)
    for _ in range(epochs):
        train_step()


if __name__ == "__main__":
    train(1000)
