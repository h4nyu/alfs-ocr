from alfs_char.data import TrainDataset
from vnet.utils import DetectionPlot
from alfs_char.store import ImageRepository
from vnet.transforms import inv_normalize
from tqdm import tqdm


def test_data() -> None:
    repo = ImageRepository()
    rows = repo.filter()
    dataset = TrainDataset(repo, rows[:1])
    for i in range(10):
        id, img, boxes, labels = dataset[0]
        plot = DetectionPlot(inv_normalize(img))
        plot.draw_boxes(boxes, color="black")
        plot.save(f"/store/tests/test-aug{i}.png")
