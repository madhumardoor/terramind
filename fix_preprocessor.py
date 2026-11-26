content = """\
from pathlib import Path
from typing import Union
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from app.core.logging_config import get_logger

logger = get_logger(__name__)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREPROCESS_PIPELINE = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ImagePreprocessor:
    def load_pil_image(self, source, bgr_input=False):
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return Image.open(path).convert("RGB")
        if isinstance(source, np.ndarray):
            if bgr_input:
                source = source[:, :, ::-1]
            return Image.fromarray(source.astype(np.uint8)).convert("RGB")
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        raise ValueError(f"Unsupported type: {type(source)}")

    def preprocess(self, source, bgr_input=False):
        pil_image = self.load_pil_image(source, bgr_input=bgr_input)
        tensor = PREPROCESS_PIPELINE(pil_image)
        return tensor.unsqueeze(0)

    def preprocess_batch(self, sources, bgr_input=False):
        tensors = []
        for source in sources:
            pil_image = self.load_pil_image(source, bgr_input=bgr_input)
            tensors.append(PREPROCESS_PIPELINE(pil_image))
        return torch.stack(tensors, dim=0)

    def get_image_info(self, source):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
        return {
            "filename": path.name,
            "width": width,
            "height": height,
            "mode": mode,
            "file_size_bytes": path.stat().st_size,
        }
"""

with open("backend/app/engines/vision/image_preprocessor.py", "w", encoding="utf-8") as f:
    f.write(content)

print("File written successfully")