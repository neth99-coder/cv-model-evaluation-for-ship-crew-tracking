"""
SSDLite320 MobileNetV3-Large person detector via torchvision.

Weights: SSDLite320_MobileNet_V3_Large_Weights.DEFAULT (COCO-pretrained).
COCO label 1 = person (torchvision uses 1-indexed labels; 0 = background).

Significantly faster than Faster R-CNN at the cost of some accuracy.
The model runs on GPU if available, otherwise CPU.
"""
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .base import BasePersonDetector

PERSON_LABEL = 1  # torchvision COCO label for "person"


class SSDMobileNetDetector(BasePersonDetector):
    model_name = "SSDLite320_MobileNetV3"

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        import torch
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )

        self.confidence_threshold = confidence_threshold
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self._model = ssdlite320_mobilenet_v3_large(weights=weights)
        self._model.eval()
        self._model.to(self._device)
        self._transforms = weights.transforms()

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self._transforms(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            outputs = self._model(tensor)[0]

        detections: list[dict[str, Any]] = []
        for box, label, score in zip(
            outputs["boxes"], outputs["labels"], outputs["scores"]
        ):
            if int(label) != PERSON_LABEL:
                continue
            if float(score) < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": round(float(score), 4),
            })
        return detections
