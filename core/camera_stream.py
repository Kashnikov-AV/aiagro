import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


class CameraStream(QThread):
    """Независимый поток захвата видео с одной камеры.

    Архитектура (аналогично popular проектам на GitHub, напр.
    QOpenCV / pyqt5-opencv-streaming примерам):
      - весь цикл cap.read() живёт внутри QThread.run(), GUI-поток не блокируется;
      - кадры конвертируются в QImage и передаются сигналом frame_ready;
      - если камера не найдена или поток оборвался — сигнал error_occurred
        emitится ОДИН раз и поток завершается (без реконнекта);
      - stop() корректно освобождает ресурсы камеры.
    """

    frame_ready = pyqtSignal(QImage)
    error_occurred = pyqtSignal(str)
    result_ready = pyqtSignal(dict)  # сводка от pipeline (total, in_roi, ...)

    def __init__(self, camera_index: int, fps_limit: float = 30.0,
                 pipeline=None, display_key: str | None = None, parent=None):
        """pipeline — VideoPipeline из core.pipeline или None (чистый поток).
        display_key — ключ ctx.result с доп. изображением для frame_ready;
        по умолчанию используется ctx.annotated."""
        super().__init__(parent)
        self.camera_index = camera_index
        self.frame_interval = 1.0 / fps_limit if fps_limit > 0 else 0.0
        self.pipeline = pipeline
        self.display_key = display_key
        self._running = False
        self._cap = None

    def run(self):
        self._running = True
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            self.error_occurred.emit(f"Камера {self.camera_index} не найдена")
            return

        # Пробное чтение: камера может «открываться», но не отдавать кадры
        ret, _ = self._cap.read()
        if not ret:
            self._cap.release()
            self._cap = None
            self.error_occurred.emit(f"Камера {self.camera_index} не отвечает")
            return

        while self._running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                if self._running:
                    self.error_occurred.emit(f"Поток камеры {self.camera_index} прерван")
                break

            if self.pipeline is not None:
                ctx = self.pipeline.run(frame)
                # что показывать: доп. изображение из result или аннотированный кадр
                img_rgb = None
                if self.display_key and ctx.result.get(self.display_key) is not None:
                    img_rgb = ctx.result[self.display_key]
                elif ctx.annotated is not None:
                    img_rgb = ctx.annotated
                if img_rgb is not None:
                    image = self._ndarray_to_qimage(img_rgb)
                else:
                    image = self._to_qimage(frame)
                if self._running:
                    self.result_ready.emit({k: v for k, v in ctx.result.items()
                                            if not isinstance(v, np.ndarray)})
            else:
                image = self._to_qimage(frame)

            if image is not None and self._running:
                self.frame_ready.emit(image)

            if self.frame_interval > 0:
                self.msleep(int(self.frame_interval * 1000))

        self._release()

    @staticmethod
    def _ndarray_to_qimage(rgb):
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w,
                      QImage.Format.Format_RGB888).copy()

    @staticmethod
    def _to_qimage(frame_bgr):
        """BGR ndarray -> QImage без копирования лишнего буфера."""
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            # keepalive на rgb через .copy() гарантирует владение данными QImage
            return QImage(rgb.data, w, h, bytes_per_line,
                          QImage.Format.Format_RGB888).copy()
        except Exception:
            return None

    @staticmethod
    def error_pixmap_placeholder(text: str) -> QImage:
        """Заглушка 'камера недоступна' (рисуется numpy-массивом, без виджетов)."""
        img = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.putText(img, text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

    def stop(self):
        """Остановка потока и освобождение камеры (потокобезопасно)."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
