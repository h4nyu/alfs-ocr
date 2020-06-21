import torch
import typing as t
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from torch import Tensor


class DetectionPlot:
    def __init__(self, figsize: t.Tuple[int, int] = (4, 4)) -> None:
        self.w, self.h = (128, 128)
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.imshow(torch.ones(self.w, self.h, 3), interpolation="nearest")

    def __del__(self) -> None:
        plt.close(self.fig)

    def save(self, path: t.Union[str, Path]) -> None:
        self.fig.savefig(path)

    def with_image(self, image: Tensor) -> None:
        if len(image.shape) == 2:
            self.ax.imshow(image, interpolation="nearest")
            self.h, self.w = image.shape
        elif len(image.shape) == 3:
            _, self.h, self.w, = image.shape
            image = image.permute(1, 2, 0)
            self.ax.imshow(image, interpolation="nearest")
        else:
            shape = image.shape
            raise ValueError(f"invald shape={shape}")

    def with_boxes(
        self,
        boxes: Tensor,
        probs: t.Optional[Tensor] = None,
        color: str = "black",
        fontsize: int = 7,
    ) -> None:
        """
        boxes: coco format
        """
        b, _ = boxes.shape
        _probs = probs if probs is not None else torch.ones((b,))
        _boxes = boxes.clone()
        for box, p in zip(_boxes, _probs):
            x0 = box[0]
            y0 = box[1]
            self.ax.text(x0, y0, f"{p:.2f}", fontsize=fontsize, color=color)
            rect = mpatches.Rectangle(
                (x0, y0),
                width=box[2],
                height=box[3],
                fill=False,
                edgecolor=color,
                linewidth=1,
            )
            self.ax.add_patch(rect)
