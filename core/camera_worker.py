import subprocess
import cv2
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class CameraWorker(QObject):
    camera_initialized = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cap = None

    def _find_camera_device(self, name_substring="Q8 HD Webcam"):
        """Ищет путь к /dev/videoX по имени камеры."""
        try:
            output = subprocess.check_output(
                ["v4l2-ctl", "--list-devices"], text=True
            )
            current_name = ""
            for line in output.splitlines():
                if not line.startswith("\t") and line.strip():
                    current_name = line.strip()
                elif line.startswith("\t"):
                    dev_path = line.strip()
                    if name_substring.lower() in current_name.lower():
                        return dev_path
        except Exception as e:
            print(f"Ошибка поиска камеры: {e}")
        return None

    @pyqtSlot()
    def initialize_camera(self):
        try:
            device = self._find_camera_device("Q8 HD Webcam") or "/dev/video1" or "/dev/video2"
            print(f"Использую камеру: {device}")
            
            self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.error_occurred.emit(f"Не удалось открыть камеру {device}")
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