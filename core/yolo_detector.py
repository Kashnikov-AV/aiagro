"""Ленивая загрузка YOLO-модели (общий экземпляр на все камеры).

Модель загружается один раз при первом обращении и переиспользуется всеми
потоками CameraStream. Ультралитиксовский predict не потокобезопасен на 100%,
поэтому инференс сериализуется простым Lock — при двух камерах это ~5-10 мс
дополнительной задержки, что приемлемо для 30 fps.

Если ultralytics не установлен или весов нет в models/ — is_available()
возвращает False, и пайплайн GOG деградирует до режима без YOLO
(обработка всего кадра, как раньше), приложение не падает.
"""

from __future__ import annotations

import threading
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_WEIGHTS = MODEL_DIR / "yolov8n.pt"


class YoloModel:
    """Потокобезопасная обёртка над ultralytics.YOLO."""

    _instance: "YoloModel | None" = None
    _init_lock = threading.Lock()

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS):
        self._model = None
        self._lock = threading.Lock()
        self._error: str | None = None
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(weights))
        except Exception as e:  # ImportError, битые веса, нет CUDA и т.п.
            self._error = str(e)

    @classmethod
    def get(cls, weights: str | Path = DEFAULT_WEIGHTS) -> "YoloModel":
        """Синглтон: одна модель на процесс."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(weights)
        return cls._instance

    def is_available(self) -> bool:
        return self._model is not None

    @property
    def error(self) -> str | None:
        return self._error

    def predict(self, frame_bgr, conf: float = 0.25, iou: float = 0.45,
                classes: list[int] | None = None, imgsz: int = 640):
        """Инференс одного кадра. Возвращает список Results (ultralytics).

        classes — фильтр по COCO-id (например [39] = "plant"), None = все.
        """
        if self._model is None:
            return []
        with self._lock:
            return self._model.predict(
                frame_bgr, conf=conf, iou=iou, classes=classes,
                imgsz=imgsz, verbose=False)
