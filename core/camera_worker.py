import cv2
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class CameraWorker(QObject):
    camera_initialized = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cap = None

    @pyqtSlot()
    def initialize_camera(self):
        """Инициализация камеры в фоновом потоке"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.error_occurred.emit("Не удалось открыть камеру")
                return

            # Проверяем, что камера действительно работает
            ret, _ = self.cap.read()
            if not ret:
                self.error_occurred.emit("Камера не отвечает")
                self.cap.release()
                self.cap = None
                return

            self.camera_initialized.emit()

        except Exception as e:
            self.error_occurred.emit(f"Ошибка инициализации: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None

    @pyqtSlot()
    def release_camera(self):
        """Освобождение камеры"""
        if self.cap:
            self.cap.release()
            self.cap = None