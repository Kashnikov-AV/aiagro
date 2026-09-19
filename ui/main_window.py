import cv2
from pathlib import Path
from PyQt6 import QtWidgets as widgets, uic, QtCore, QtGui
from PyQt6.QtCore import Qt, QTimer, QEvent, pyqtSignal, QThread, QMetaObject
from PyQt6.QtGui import QIcon

from core.camera_worker import CameraWorker
from core.plant_detector import PlantDetector
from core.valve_controller import ValveController, CentralStripDetector
from core.detector_factory import DetectorFactory
from config.settings import check_model_availability

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

        # Инициализация компонентов (будет создана фабрикой при выборе режима)
        self.detector = None
        self.current_mode = None
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
        self.btnExpertMode.clicked.connect(self.on_expert_mode_click)
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
        """Обработка клика на режим Green-on-Green"""
        print('Режим "green on green"')
        
        # Проверка доступности модели YOLO
        if not check_model_availability():
            QtWidgets.QMessageBox.warning(
                self,
                "Модель не найдена",
                "Модель для детекции сорняков не найдена.\n"
                "Пожалуйста, обучите модель используя ml/train_weed_detector.ipynb"
            )
            return
        
        # Инициализация детектора через фабрику
        try:
            from config.settings import WEED_MODEL_PATH
            self.detector = DetectorFactory.create('green_on_green', model_path=str(WEED_MODEL_PATH))
            self.current_mode = 'green_on_green'
            
            # Запуск видеорежима
            self.stackedWidget.setCurrentWidget(self.loadingPage)
            self.loading_movie.start()
            self.start_initialization.emit()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Ошибка инициализации",
                f"Не удалось инициализировать режим Green-on-Green:\n{str(e)}"
            )
            print(f"Error initializing green_on_green mode: {e}")

    def on_green_brown_click(self):
        """Обработка клика на стандартный режим (зелёное на коричневом)"""
        # Инициализация стандартного детектора через фабрику
        self.detector = DetectorFactory.create('standard')
        self.current_mode = 'standard'
        
        self.stackedWidget.setCurrentWidget(self.loadingPage)
        self.loading_movie.start()
        self.start_initialization.emit()

    def on_expert_mode_click(self):
        """Обработка клика на экспертный режим"""
        print('Экспертный режим - Настройка Green on Brown')
        
        # Инициализация детектора для экспертного режима (аналогично standard)
        self.detector = DetectorFactory.create('standard')
        self.current_mode = 'expert'
        
        self.stackedWidget.setCurrentWidget(self.expertPage)
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
        """Обновление кадра с учётом текущего режима"""
        if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        try:
            # Обработка в зависимости от режима
            if self.current_mode == 'green_on_green':
                self._update_frame_green_on_green(frame)
            elif self.current_mode == 'expert':
                self._update_frame_expert(frame)
            else:
                self._update_frame_standard(frame)
                
        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for label in [self.videoLabel1, self.videoLabel2, self.videoLabel3, self.videoLabel4]:
                self._update_label(label, rgb_frame)
    
    def _update_frame_expert(self, frame):
        """Обработка кадра в экспертном режиме (аналогично standard, но вывод в один виджет)"""
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

        # В экспертном режиме показываем только итоговый кадр с разметкой
        self._update_label(self.expertVideoLabel, bboxes)

        status = "ОТКРЫТ" if self.valve_controller.is_valve_open() else "ЗАКРЫТ"
        print(f"Экспертный режим | Клапан: {status} | Больших растений: {len(large_plants)} | Всего: {plant_count}")
    
    def _update_frame_standard(self, frame):
        """Обработка кадра в стандартном режиме"""
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
    
    def _update_frame_green_on_green(self, frame):
        """Обработка кадра в режиме Green-on-Green"""
        # Детектор возвращает: (кадр с разметкой, кол-во сорняков, кол-во культур)
        result_frame, weed_count, crop_count = self.detector.detect(frame)
        
        # Конвертируем из RGB в BGR если нужно (YOLO возвращает RGB)
        if len(result_frame.shape) == 3 and result_frame.shape[2] == 3:
            # Проверяем формат - если RGB, конвертируем в BGR для OpenCV
            # Проверка по первому пикселю (в RGB зелёный канал должен быть ярче)
            test_pixel = result_frame[0, 0]
            if test_pixel[1] > test_pixel[0] and test_pixel[1] > test_pixel[2]:
                # Скорее всего RGB, конвертируем
                result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
            else:
                result_frame_bgr = result_frame
        else:
            result_frame_bgr = result_frame
        
        # Для режима green_on_green показываем один итоговый кадр во всех окнах
        # или можно распределить по окнам разную информацию
        self._update_label_batch([
            (self.videoLabel1, result_frame_bgr),      # Итоговый кадр с разметкой
            (self.videoLabel2, result_frame_bgr),      # Дублируем (можно заменить на промежуточные данные)
            (self.videoLabel3, result_frame_bgr),      # Дублируем
            (self.videoLabel4, result_frame_bgr)       # Дублируем
        ])
        
        print(f"Green-on-Green | Сорняки: {weed_count} | Культуры: {crop_count}")

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
        binary = self.detector.apply_otsu(index)
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

        # Возвращаемся в меню из любого режима
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