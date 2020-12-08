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
        plot = DetectionPlot(w=1024,h=1024)
        plot.with_image(img)
        plot.with_yolo_boxes(boxes)
        plot.save(f"/store/tests/test-aug{i}.png")
