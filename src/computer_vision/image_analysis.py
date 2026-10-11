"""Optional image analysis (auxiliary evidence only; NOT a fake-review classifier).

OpenCV: validation, dimensions, blur, brightness, quality heuristic, perceptual hash.
YOLO (opt-in via TRUSTLENS_ENABLE_YOLO=1, needs `ultralytics`; weights may download on first use):
object detection. Duplicate detection only compares against images seen by this running process.
"""

from __future__ import annotations

import os
import threading

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_DIMENSION = 12000
HASH_DUPLICATE_DISTANCE = 5  # Hamming distance on a 64-bit dHash
MAGIC = {b"\xff\xd8\xff": "jpeg", b"\x89PNG\r\n\x1a\n": "png", b"RIFF": "webp"}

try:
    import cv2
    import numpy as np
except ImportError:  # core TrustLens works without CV dependencies
    cv2 = None
    np = None


def cv_available() -> bool:
    return cv2 is not None


def sniff_format(data: bytes):
    for magic, name in MAGIC.items():
        if data.startswith(magic):
            if name == "webp" and data[8:12] != b"WEBP":
                continue
            return name
    return None


def decode_image(data: bytes):
    """Validate and decode. Raises ValueError with a safe message for bad input."""
    if not cv_available():
        raise RuntimeError("OpenCV is not installed")
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image must be non-empty and at most {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )
    if sniff_format(data) is None:
        raise ValueError("Unsupported image type; use JPEG, PNG or WebP.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image is corrupted or cannot be decoded.")
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        raise ValueError("Image dimensions are too large.")
    return img


def dhash(img) -> int:
    """64-bit difference hash (robust to resizing and re-compression)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class ImageHashIndex:
    """In-memory perceptual-hash index (process-local; not persisted)."""

    def __init__(self):
        self._items: list[int] = []
        self._lock = threading.Lock()

    def check_and_add(self, h: int, max_distance: int = HASH_DUPLICATE_DISTANCE) -> bool:
        with self._lock:
            dup = any(hamming(h, x) <= max_distance for x in self._items)
            self._items.append(h)
            return dup

    def __len__(self):
        return len(self._items)


_yolo = None


def detect_objects(img) -> dict:
    """YOLO detection if explicitly enabled and installed; otherwise reports unavailability."""
    global _yolo
    if os.getenv("TRUSTLENS_ENABLE_YOLO") != "1":
        return {
            "available": False,
            "reason": "YOLO disabled (set TRUSTLENS_ENABLE_YOLO=1)",
            "objects": [],
        }
    try:
        if _yolo is None:
            from ultralytics import YOLO

            _yolo = YOLO(os.getenv("TRUSTLENS_YOLO_WEIGHTS", "yolov8n.pt"))
        res = _yolo(img, verbose=False)[0]
        objs = [
            {"label": res.names[int(c)], "confidence": float(p)}
            for c, p in zip(res.boxes.cls, res.boxes.conf)
        ]
        return {"available": True, "objects": objs}
    except ImportError:
        return {
            "available": False,
            "reason": "ultralytics is not installed (pip install -r requirements-cv.txt)",
            "objects": [],
        }
    except Exception as e:  # weights download/inference problems must not break text analysis
        return {"available": False, "reason": f"YOLO failed: {type(e).__name__}", "objects": []}


def analyze_image(data: bytes, index: ImageHashIndex | None = None, run_yolo: bool = False) -> dict:
    img = decode_image(data)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    anomalies = []
    if min(h, w) < 224:
        anomalies.append("low_resolution")
    if blur < 50:
        anomalies.append("blurry")
    if brightness < 40:
        anomalies.append("very_dark")
    if brightness > 220:
        anomalies.append("very_bright")
    # Quality heuristic in [0,1]: mean of sharpness, exposure and resolution sub-scores (not validated).
    quality = (
        min(1.0, blur / 100) + (1.0 if 40 <= brightness <= 220 else 0.3) + min(1.0, min(h, w) / 224)
    ) / 3
    ph = dhash(img)
    dup = index.check_and_add(ph) if index is not None else False
    if dup:
        anomalies.append("duplicate_image")
    det = (
        detect_objects(img)
        if run_yolo
        else {"available": False, "reason": "not requested", "objects": []}
    )
    return {
        "width": w,
        "height": h,
        "blur_score": blur,
        "brightness": brightness,
        "image_quality_score": round(quality, 3),
        "perceptual_hash": f"{ph:016x}",
        "duplicate_image_flag": dup,
        "detected_objects": det["objects"],
        "object_count": len(det["objects"]),
        "object_detection": {k: det[k] for k in det if k != "objects"},
        "image_anomaly_signals": anomalies,
    }
