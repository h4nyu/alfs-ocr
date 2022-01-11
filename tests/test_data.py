import pytest
from typing import Any
from alfs_char.data import TrainDataset, TrainTransform, Transform, collate_fn
from alfs_char.store import ImageRepository
from torch.utils.data import DataLoader
from vision_tools.utils import batch_draw
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("/app/runs/test")


def test_data() -> None:
    repo = ImageRepository()
    rows = repo.filter()
    dataset = TrainDataset(repo, rows, TrainTransform(512))
    loader_iter = iter(DataLoader(dataset, batch_size=8, collate_fn=collate_fn))
    batch = next(loader_iter)
    plot = batch_draw(**batch)
    writer.add_image("aug", plot, 0)
    writer.flush()
