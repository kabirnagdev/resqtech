"""
YOLO person detector + ByteTrack association.

Detection and tracking are fused into a single call here because that is
Ultralytics' officially supported, version-stable API
(``model.track(..., tracker="bytetrack.yaml")``). Everything *downstream*
of a per-frame detection list -- track lifecycle, first/last-seen
bookkeeping, enter/exit events -- lives in ``tracker.py`` instead, which
only depends on the plain ``Detection`` objects below. That boundary is the
extension point for thermal fusion later: a future ``ThermalDetector``
just needs to produce the same ``Detection`` list (optionally merged with
the RGB list in a fusion step) and it can feed the exact same
``TrackManager``.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from ultralytics import YOLO

from . import config


@dataclass
class Detection:
    track_id: Optional[int]
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords
    confidence: float
    class_id: int = config.PERSON_CLASS_ID


class PersonDetector:
    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        model_name: str = config.MODEL_NAME,
        device: str = config.DEVICE,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
    ):
        # Reuse an existing weights file (models/<name>) if present so the
        # app works offline after the first run; otherwise Ultralytics
        # downloads the named model and caches it there.
        weights = model_path if os.path.exists(model_path) else model_name
        self.model = YOLO(weights)
        self.device = self._resolve_device(device)
        self.confidence = confidence
        self.iou = iou

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    def track(self, frame) -> tuple[list[Detection], float]:
        """Run detection + ByteTrack association on a single BGR frame.

        Returns ``(detections, inference_latency_ms)``.
        """
        start = time.perf_counter()
        results = self.model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            classes=[config.PERSON_CLASS_ID],
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        detections: list[Detection] = []
        result = results[0] if results else None
        if result is not None and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            ids = boxes.id
            xyxy = boxes.xyxy
            confs = boxes.conf
            for i in range(len(boxes)):
                track_id = int(ids[i].item()) if ids is not None else None
                x1, y1, x2, y2 = (float(v) for v in xyxy[i].tolist())
                conf = float(confs[i].item())
                detections.append(
                    Detection(track_id=track_id, bbox=(x1, y1, x2, y2), confidence=conf)
                )
        return detections, latency_ms

    def reset_tracker_state(self) -> None:
        """Clear ByteTrack's internal state, e.g. after a camera reconnect
        so stale track IDs from before the gap aren't reused."""
        try:
            predictor = self.model.predictor
            if predictor is not None and getattr(predictor, "trackers", None):
                for t in predictor.trackers:
                    t.reset()
        except Exception:
            pass
