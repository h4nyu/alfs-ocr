from alfs_char.data import TrainDataset
from object_detection.utils import DetectionPlot
from tqdm import tqdm


def test_data() -> None:
    dataset = TrainDataset("http://yakiniku.mydns.jp:2030/")
    for j in range(2):
        for i in tqdm(range(len(dataset))):
            id, img, boxes, labels = dataset[i]
        # print(img)
        # plot = DetectionPlot(w=1024,h=1024)
        # plot.with_image(img)
        # plot.with_yolo_boxes(boxes)
        # plot.save(f"/store/tests/{id}.png")
