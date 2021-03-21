from vnet import Boxes, Confidences
from PIL import Image as PILImage

def detect(img: PILImage) -> tuple[Boxes, Confidences]:
    ...
