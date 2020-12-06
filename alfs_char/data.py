from torch.utils.data import Dataset
from object_detection.entities import (
    TrainSample,
    ImageId,
    Image,
)


class TrainDataset(Dataset):
    def __init__(
        self,
    ) -> None:
        ...

    def __getitem__(self, idx: int) -> TrainSample:
        ...

    def __len__(self) -> int:
        ...
