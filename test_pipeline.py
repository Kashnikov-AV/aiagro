"""Тесты пайплайнов GOB/GOG (core/pipeline.py). Запуск: pytest test_pipeline.py"""
import numpy as np
import cv2

from core.pipeline import (
    PipelineContext, build_gob_pipeline, build_gog_pipeline,
    YoloCropStage, WeedLabelStage,
)


def _synthetic_frame(h=480, w=640):
    """Зелёные 'растения' на коричневом фоне."""
    frame = np.full((h, w, 3), (40, 70, 110), np.uint8)  # BGR: коричневый
    for cx, cy in [(120, 200), (320, 300), (500, 150)]:
        cv2.circle(frame, (cx, cy), 30, (30, 160, 40), -1)  # зелёный круг
    return frame


def test_gob_pipeline_runs():
    pipe = build_gob_pipeline()
    out = pipe.run(_synthetic_frame())
    assert out.annotated is not None
    assert out.binary is not None and out.bitmap is not None


def test_gog_pipeline_marks_weeds():
    pipe = build_gog_pipeline(use_yolo=False)
    out = pipe.run(_synthetic_frame())
    assert out.result.get("weeds", 0) >= 1
    assert all(p["label"] == "weed" for p in out.plants)
    assert out.annotated is not None


def test_yolo_stage_degrades_without_model():
    stage = YoloCropStage(weights="__no_such_file__.pt")
    ctx = PipelineContext(frame=_synthetic_frame())
    out = stage.process(ctx)  # не должно упасть
    assert "yolo" in out.result  # записан статус недоступности
    assert out.crops == []


def test_weed_label_stage_filters_noise():
    ctx = PipelineContext(frame=_synthetic_frame())
    # одно растение-шум (площадь 1 px) и одно настоящее
    ctx.plants = [{"area": 1.0, "bbox": (0, 0, 1, 1), "size": "s"},
                  {"area": 900.0, "bbox": (10, 10, 40, 40), "size": "m"}]
    stage = WeedLabelStage(min_area=50)
    out = stage.process(ctx)
    assert out.result["weeds"] == 1
    assert out.plants[0]["label"] == "weed"
