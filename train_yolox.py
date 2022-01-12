import os
import torch_optimizer as optim
import torch
from torch import Tensor
from omegaconf import OmegaConf
from torch.utils.data import Subset, DataLoader
from vision_tools.utils import seed_everything, Checkpoint, ToDevice
from vision_tools.yolox import YOLOX, Criterion, Inference
from alfs_char.yolox import (
    get_model,
    get_criterion,
    get_checkpoint,
    get_writer,
)
from vision_tools.meter import MeanReduceDict
from vision_tools.step import TrainStep, EvalStep
from vision_tools.interface import TrainBatch
from alfs_char.store import ImageRepository
from alfs_char.data import (
    TrainDataset,
    TrainTransform,
    Transform,
    collate_fn,
)
from alfs_char.metric import Metric
from tqdm import tqdm


def main() -> None:
    seed_everything()
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config/yolox.yaml"))
    checkpoint = get_checkpoint(cfg)
    writer = get_writer(cfg)
    model = get_model(cfg)
    criterion = get_criterion(cfg)
    optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer)

    checkpoint.load_if_exists(model=model, optimizer=optimizer, device=cfg.device)

    repo = ImageRepository()
    rows = repo.filter()
    image_count = len(rows)
    train_dataset = TrainDataset(
        repo,
        rows[: int(image_count * 0.8)],
        transform=TrainTransform(cfg.image_size),
    )
    val_dataset = TrainDataset(
        repo,
        rows[int(image_count * 0.8) :],
        transform=Transform(cfg.image_size),
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        **cfg.train_loader,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        **cfg.val_loader,
    )
    to_device = ToDevice(cfg.device)

    train_step = TrainStep[YOLOX, TrainBatch](
        to_device=to_device,
        criterion=criterion,
        optimizer=optimizer,
        loader=train_loader,
        meter=MeanReduceDict(),
        writer=writer,
        checkpoint=checkpoint,
        use_amp=cfg.use_amp,
    )
    metric = Metric()

    eval_step = EvalStep[YOLOX, TrainBatch](
        to_device=to_device,
        loader=val_loader,
        metric=metric,
        writer=writer,
        inference=Inference(),
        checkpoint=checkpoint,
    )

    for epoch in range(cfg.num_epochs):
        train_step(model, epoch)
        eval_step(model, epoch)


if __name__ == "__main__":
    main()
