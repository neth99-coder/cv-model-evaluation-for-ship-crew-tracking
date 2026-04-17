"""
Faster R-CNN (ResNet-50 FPN v2) person detector via torchvision.

Weights: FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT (COCO-pretrained).
COCO label 1 = person (torchvision models use 1-indexed labels where 0 is background).

The model runs on GPU if available, otherwise CPU.
"""
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .base import BasePersonDetector

PERSON_LABEL = 1  # torchvision COCO label for "person" (1-indexed, 0 = background)


class FasterRCNNDetector(BasePersonDetector):
    model_name = "FasterRCNN_ResNet50_FPN_v2"

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        import torch
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights,
        )

        self.confidence_threshold = confidence_threshold
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self._model = fasterrcnn_resnet50_fpn_v2(weights=weights)
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
