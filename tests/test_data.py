from alfs_char.data import TrainDataset
from object_detection.utils import DetectionPlot
from alfs_char.store import ImageRepository
from tqdm import tqdm


def test_data() -> None:
    repo = ImageRepository()
    rows = repo.filter()
    dataset = TrainDataset(repo, rows[:1])
    for i in range(10):
        id, img, boxes, labels = dataset[0]
        plot = DetectionPlot(img)
        plot.draw_boxes(boxes)
        plot.save(f"/store/tests/test-aug{i}.png")


def test_predict() -> None:
    repo = ImageRepository()
    rows = repo.filter()
    row = rows[0]
    imageId = row["id"]
    boxes = [dict(x0=0.1, y0=0.1, x1=0.2, y1=0.2, imageId=imageId)]
    repo.predict(imageId, boxes)
