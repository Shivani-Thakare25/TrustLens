import cv2
import numpy as np
import pytest

from src.computer_vision import image_analysis as ia

pytestmark = pytest.mark.skipif(not ia.cv_available(), reason="OpenCV not installed")


def make_img(seed=0, size=320):
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 128, np.uint8)
    for _ in range(14):
        x0, y0 = (int(v) for v in rng.integers(0, size - 120, 2))
        w, h = (int(v) for v in rng.integers(30, 110, 2))
        cv2.rectangle(
            img, (x0, y0), (x0 + w, y0 + h), tuple(int(c) for c in rng.integers(0, 255, 3)), -1
        )
    return img


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_good_image_metrics():
    r = ia.analyze_image(png(make_img()))
    assert (r["width"], r["height"]) == (320, 320)
    assert r["image_anomaly_signals"] == [] and r["image_quality_score"] > 0.8
    assert r["object_detection"]["available"] is False


def test_anomaly_flags():
    assert (
        "blurry"
        in ia.analyze_image(png(cv2.GaussianBlur(make_img(), (41, 41), 0)))["image_anomaly_signals"]
    )
    assert "very_dark" in ia.analyze_image(png(make_img() // 10))["image_anomaly_signals"]
    assert (
        "low_resolution"
        in ia.analyze_image(png(make_img(size=320)[:100, :100]))["image_anomaly_signals"]
    )
    assert (
        ia.analyze_image(png(make_img() // 10))["image_quality_score"]
        < ia.analyze_image(png(make_img()))["image_quality_score"]
    )


def test_duplicate_detection_survives_resize_and_recompression():
    idx = ia.ImageHashIndex()
    assert ia.analyze_image(png(make_img(0)), idx)["duplicate_image_flag"] is False
    resized = cv2.resize(make_img(0), (256, 256))
    jpg = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    r = ia.analyze_image(jpg, idx)
    assert r["duplicate_image_flag"] is True and "duplicate_image" in r["image_anomaly_signals"]
    assert ia.analyze_image(png(make_img(1)), idx)["duplicate_image_flag"] is False
    assert len(idx) == 3


@pytest.mark.parametrize(
    "data,msg",
    [
        (b"", "non-empty"),
        (b"GIF89a" + b"0" * 50, "Unsupported"),
        (b"just some text", "Unsupported"),
        (b"\x89PNG\r\n\x1a\n" + b"junk" * 20, "corrupted"),
        (b"\xff\xd8\xff\xe0" + b"0" * 20, "corrupted"),
        (b"\x89PNG\r\n\x1a\n" + b"0" * (5 * 1024 * 1024 + 1), "at most"),
    ],
)
def test_invalid_images_rejected(data, msg):
    with pytest.raises(ValueError, match=msg):
        ia.analyze_image(data)


def test_yolo_disabled_by_default_and_graceful_without_ultralytics(monkeypatch):
    img = make_img()
    assert ia.detect_objects(img)["available"] is False
    monkeypatch.setenv("TRUSTLENS_ENABLE_YOLO", "1")
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", None)
    r = ia.detect_objects(img)
    assert r["available"] is False and "ultralytics" in r["reason"]


def test_yolo_path_with_stub(monkeypatch):
    class Boxes:
        cls, conf = [0, 1], [0.9, 0.5]

    class Res:
        names = {0: "cell phone", 1: "cup"}
        boxes = Boxes()

    class FakeYOLO:
        def __init__(self, w):
            pass

        def __call__(self, img, verbose=False):
            return [Res()]

    monkeypatch.setenv("TRUSTLENS_ENABLE_YOLO", "1")
    monkeypatch.setattr(ia, "_yolo", None)
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", type("M", (), {"YOLO": FakeYOLO}))
    r = ia.analyze_image(png(make_img()), run_yolo=True)
    assert r["object_count"] == 2 and r["detected_objects"][0]["label"] == "cell phone"
    monkeypatch.setattr(ia, "_yolo", None)
