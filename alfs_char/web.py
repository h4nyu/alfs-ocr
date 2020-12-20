from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import List, Any
import torch
import numpy as np
from PIL import Image as PILImage
import base64
from torchvision.transforms import ToPILImage
from io import BytesIO
import base64
import asyncio
from alfs_char.train import model, model_loader, to_boxes
from alfs_char.data import test_transforms
from alfs_char import config

app = Flask(__name__)
CORS(app)

model_loader.load_if_needed(model)
device = torch.device("cuda")

@torch.no_grad()
def detect() -> Any:
    image = np.array(
        PILImage.open(BytesIO(base64.b64decode(request.json['data']))).convert("RGB")
    )
    img_tensor = test_transforms(image=image, labels=[], bboxes=[])["image"]
    batch = torch.stack([img_tensor / 255]).to(device)
    netout = model(batch)
    boxes_list, scores_list = to_boxes(netout)
    out_boxes = boxes_list[0] / config.image_size
    out_scores = scores_list[0]
    out_img = ToPILImage()(batch[0].cpu())
    buffer = BytesIO()
    out_img.save(buffer, format="JPEG")
    return jsonify(
        image=base64.b64encode(buffer.getvalue()).decode('utf-8'),
        boxes=out_boxes.tolist(),
        scores=out_scores.tolist(),
    )


app.route("/api/upload-image", methods=["POST"])(detect)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
