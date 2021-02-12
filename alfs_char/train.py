import torch
from torch.utils.data import DataLoader
from object_detection.models.backbones.effnet import (
    EfficientNetBackbone,
)
from object_detection.models.effidet import (
    collate_fn,
    EfficientDet,
    Trainer,
    Criterion,
    Visualize,
    ToBoxes,
    Anchors,
)
from object_detection.model_loader import (
    ModelLoader,
    BestWatcher,
)
from alfs_char.data import TrainDataset
from object_detection.metrics import MeanPrecition
from alfs_char import config
from alfs_char.store import ImageRepository
from alfs_char.transforms import RandomDilateErode, RandomLayout, RandomRuledLines

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
backbone = EfficientNetBackbone(
    config.backbone_idx, out_channels=config.channels, pretrained=True
)
anchors = Anchors(
    size=config.anchor_size,
    ratios=config.anchor_ratios,
    scales=config.anchor_scales,
)
model = EfficientDet(
    num_classes=1,
    channels=config.channels,
    backbone=backbone,
    anchors=anchors,
    out_ids=config.out_ids,
    box_depth=config.depth,
)
model_loader = ModelLoader(
    out_dir=config.out_dir,
    key=config.metric[0],
    best_watcher=BestWatcher(mode=config.metric[1]),
)
criterion = Criterion(
    topk=config.topk,
    box_weight=config.box_weight,
)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
visualize = Visualize(config.out_dir, "test", limit=6)
get_score = MeanPrecition()
to_boxes = ToBoxes(
    confidence_threshold=config.confidence_threshold,
)
trainer = Trainer(
    model,
    DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        shuffle=True,
    ),
    DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size * 2,
        shuffle=False,
    ),
    model_loader=model_loader,
    optimizer=optimizer,
    visualize=visualize,
    criterion=criterion,
    get_score=get_score,
    device="cuda",
    to_boxes=to_boxes,
)


if __name__ == "__main__":
    trainer(10000)
