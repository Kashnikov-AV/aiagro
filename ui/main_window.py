import cv2
from pathlib import Path
from PyQt6 import QtWidgets as widgets, uic, QtCore, QtGui
from PyQt6.QtCore import Qt, QTimer, QEvent, pyqtSignal, QThread, QMetaObject
from PyQt6.QtGui import QIcon

from core.camera_worker import CameraWorker
from core.plant_detector import PlantDetector
from core.valve_controller import ValveController, CentralStripDetector

# Путь к файлу design.ui
UI_FILE_PATH = Path(__file__).parent.parent / "design.ui"
Design, _ = uic.loadUiType(str(UI_FILE_PATH))


class MainWindow(widgets.QMainWindow, Design):
    start_initialization = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.project_root = Path(__file__).parent.parent

        # Настройка UI
        self._setup_ui()

        # Инициализация компонентов
        self.detector = PlantDetector(index_type='exg', downscale_factor=0.5)
        self.valve_controller = ValveController(gpio_pin=7, valve_open_time=0.5, debug=True)
        self.strip_detector = CentralStripDetector(strip_height_percent=0.3, min_plant_area=2000)

        # Настройка камеры
        self._setup_camera_thread()

        # Подключение сигналов
        self._connect_signals()

        # Запуск таймера курсора
        self.reset_cursor_timer()

    def _setup_ui(self):
        self.showFullScreen()
        self.setMouseTracking(True)

        loading_path = self.project_root / "icons" / "loading.gif"
        reject_path = self.project_root / "icons" / "reject.png"

        self.loading_movie = QtGui.QMovie(str(loading_path))
        self.loadingLabel.setMovie(self.loading_movie)
        self.loading_movie.setScaledSize(QtCore.QSize(80, 80))

        self.exitButton.setIcon(QIcon(str(reject_path)))
        self.exitButton.setIconSize(QtCore.QSize(40, 40))
        self.exitButton.setText("")

        self.installEventFilter(self)
        self.videoPage.installEventFilter(self)

        # Инициализация таймера курсора
        self.inactivity_timer = QTimer()
        self.inactivity_timer.setSingleShot(True)
        self.inactivity_timer.timeout.connect(self.hide_cursor)
        self.inactivity_timeout = 3000

        self.showFullScreen()
        self.setMouseTracking(True)

    def _setup_camera_thread(self):
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker()
        self.camera_worker.moveToThread(self.camera_thread)

        # Подключение сигналов
        self.start_initialization.connect(self.camera_worker.initialize_camera)
        self.camera_worker.camera_initialized.connect(self.on_camera_ready)
        self.camera_worker.error_occurred.connect(self.on_camera_error)

        self.camera_thread.start()

    def _connect_signals(self):
        self.exitButton.clicked.connect(self.close)
        self.btnGreenGreen.clicked.connect(self.on_green_green_click)
        self.btnGreenBrown.clicked.connect(self.on_green_brown_click)
        self.btnBackToMenu.clicked.connect(self.stop_video_mode)

    def reset_cursor_timer(self):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.inactivity_timer.stop()
        self.inactivity_timer.start(self.inactivity_timeout)

    def hide_cursor(self):
        self.setCursor(Qt.CursorShape.BlankCursor)

    def eventFilter(self, obj, event):
        activity_events = [
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.Wheel,
        ]

        if event.type() in activity_events:
            self.reset_cursor_timer()

        return super().eventFilter(obj, event)

    def on_green_green_click(self):
        print('Режим "green on green"')

    def on_green_brown_click(self):
        self.stackedWidget.setCurrentWidget(self.loadingPage)
        self.loading_movie.start()
        self.start_initialization.emit()

    def on_camera_ready(self):
        self.cap = self.camera_worker.cap
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)
        self.stackedWidget.setCurrentWidget(self.videoPage)
        self.loading_movie.stop()

    def on_camera_error(self, error_msg):
        print(error_msg)
        self.loadingText.setText("Ошибка подключения!")
        self.loading_movie.stop()
        QTimer.singleShot(2000, self.return_to_menu)

    def return_to_menu(self):
        self.loadingText.setText("Идёт загрузка...")
        self.loading_movie.start()
        self.stackedWidget.setCurrentWidget(self.menuPage)
        self.reset_cursor_timer()

    def update_frame(self):
        if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        try:
            original, index_map, bitmap, bboxes, plant_count = self.detector.process_frame(frame)

            # Получаем все контуры для проверки полосы
            _, _, _, s_contours, m_contours, l_contours = self._get_plant_contours(frame)
            all_contours = s_contours + m_contours + l_contours

            plants_in_strip, strip_center, strip_bounds, plants_list = \
                self.strip_detector.check_plants_in_strip(all_contours, original.shape)

            # Открываем клапан только для больших растений
            large_plants = [p for p in plants_list if p['area'] > 2000]
            if large_plants and not self.valve_controller.is_valve_open():
                self.valve_controller.open_valve()

            if strip_bounds:
                strip_top, strip_bottom = strip_bounds
                bboxes = self.strip_detector.draw_central_strip(
                    bboxes, strip_top, strip_bottom, len(large_plants) > 0
                )

            self._update_label_batch([
                (self.videoLabel1, original),
                (self.videoLabel2, index_map),
                (self.videoLabel3, bitmap),
                (self.videoLabel4, bboxes)
            ])

            status = "ОТКРЫТ" if self.valve_controller.is_valve_open() else "ЗАКРЫТ"
            print(f"Клапан: {status} | Больших растений: {len(large_plants)} | Всего: {plant_count}")

        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for label in [self.videoLabel1, self.videoLabel2, self.videoLabel3, self.videoLabel4]:
                self._update_label(label, rgb_frame)

    def _get_plant_contours(self, frame):
        import cv2
        import numpy as np

        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.detector.downscale_factor < 1.0:
            h, w = original_rgb.shape[:2]
            new_size = (int(w * self.detector.downscale_factor), int(h * self.detector.downscale_factor))
            frame_small = cv2.resize(original_rgb, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = original_rgb

        index = self.detector.calculate_index(frame_small)
        binary = self.detector.apply_otsu(index, manual_threshold=128)
        bitmap = self.detector.morph_processing(binary)

        s_contours, m_contours, l_contours = self.detector.detect_plants(bitmap, frame_small.shape[:2])

        if self.detector.downscale_factor < 1.0:
            scale = 1.0 / self.detector.downscale_factor
            s_contours = [(cnt * scale).astype(np.int32) for cnt in s_contours]
            m_contours = [(cnt * scale).astype(np.int32) for cnt in m_contours]
            l_contours = [(cnt * scale).astype(np.int32) for cnt in l_contours]

        return original_rgb, index, bitmap, s_contours, m_contours, l_contours

    def _update_label_batch(self, label_frame_pairs):
        for label, frame in label_frame_pairs:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
            label.setPixmap(scaled_pixmap)

    def _update_label(self, label, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def stop_video_mode(self):
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()
            self.timer = None

        if hasattr(self, 'cap') and self.cap:
            QMetaObject.invokeMethod(
                self.camera_worker,
                "release_camera",
                Qt.ConnectionType.QueuedConnection
            )
            self.cap = None

        self.stackedWidget.setCurrentWidget(self.menuPage)
        self.reset_cursor_timer()

    def closeEvent(self, event):
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()

        if hasattr(self, 'cap') and self.cap:
            QMetaObject.invokeMethod(
                self.camera_worker,
                "release_camera",
                Qt.ConnectionType.BlockingQueuedConnection
            )

        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.quit()
            self.camera_thread.wait()

        event.accept()

    def keyPressEvent(self, event):
        self.reset_cursor_timer()
        if event.key() == Qt.Key.Key_Escape:
            if self.stackedWidget.currentWidget() == self.videoPage:
                self.stop_video_mode()
            else:
                self.close()
        elif event.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        super().keyPressEvent(event)