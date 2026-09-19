"""
GreenOnGreenDetector - Детектор для режима "зелёное на зелёном"
Использует конвейерную обработку: YOLO (сорняки) + вегетационные индексы (культуры)
Реализует паттерн Pipeline для гибкости и расширяемости
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

from config.settings import (
    VISUALIZATION_CONFIG,
    PLANT_DETECTOR_CONFIG,
    PERFORMANCE_CONFIG
)
from core.plant_detector import PlantDetector
from ml.weed_detector import WeedDetector


class ProcessingStep(ABC):
    """Базовый класс для этапа обработки в конвейере"""
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка данных
        
        Args:
            data: Словарь с данными от предыдущих этапов
            
        Returns:
            Обновлённый словарь с данными
        """
        pass


class WeedDetectionStep(ProcessingStep):
    """Этап 1: Детекция сорняков через YOLO"""
    
    def __init__(self, weed_detector: WeedDetector):
        self.weed_detector = weed_detector
        self.frame_counter = 0
        self.skip_frames = PERFORMANCE_CONFIG.get('yolo_skip_frames', 3)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        frame = data['frame']
        self.frame_counter += 1
        
        # Пропускаем кадры для производительности
        if self.frame_counter % (self.skip_frames + 1) != 0:
            # Возвращаем предыдущие результаты если они есть
            data['weed_detections'] = data.get('last_weed_detections', [])
            data['weed_mask'] = data.get('last_weed_mask', np.zeros(frame.shape[:2], dtype=np.uint8))
            return data
        
        # Детекция сорняков
        detections, mask = self.weed_detector.detect(frame)
        
        data['weed_detections'] = detections
        data['weed_mask'] = mask
        data['last_weed_detections'] = detections
        data['last_weed_mask'] = mask
        
        return data


class MaskingStep(ProcessingStep):
    """Этап 2: Создание маски для культур (инвертированная маска сорняков)"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        weed_mask = data.get('weed_mask')
        
        if weed_mask is None or weed_mask.size == 0:
            data['crop_mask'] = np.ones(data['frame'].shape[:2], dtype=np.uint8) * 255
            return data
        
        # Инвертируем маску: где были сорняки - теперь чёрные (игнорируем)
        inverted_mask = cv2.bitwise_not(weed_mask)
        
        # Дополнительная эрозия для расширения зоны исключения вокруг сорняков
        kernel = np.ones((5, 5), np.uint8)
        inverted_mask = cv2.erode(inverted_mask, kernel, iterations=2)
        
        data['crop_mask'] = inverted_mask
        return data


class CropDetectionStep(ProcessingStep):
    """Этап 3: Детекция культур через вегетационные индексы с учётом маски"""
    
    def __init__(self, plant_detector: PlantDetector):
        self.plant_detector = plant_detector
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        frame = data['frame']
        crop_mask = data.get('crop_mask')
        
        if crop_mask is None:
            # Если маски нет, используем стандартную детекцию
            _, _, _, bboxes_img, total_count = self.plant_detector.process_frame(frame)
            data['crop_contours'] = {'small': [], 'medium': [], 'large': []}
            data['crop_bboxes_img'] = bboxes_img
            data['crop_count'] = total_count
            return data
        
        # Применяем маску к кадру перед детекцией
        masked_frame = cv2.bitwise_and(frame, frame, mask=crop_mask)
        
        # Модифицируем plant_detector для работы с маской
        # Временное изменение порога для работы с замаскированным изображением
        original_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        
        # Вычисляем вегетационный индекс
        index = self.plant_detector.calculate_index(original_rgb)
        
        # Применяем маску к индексу (обнуляем области сорняков)
        index_masked = index.copy()
        index_masked[crop_mask == 0] = -1  # Устанавливаем минимальное значение
        
        # Пороговая обработка
        binary = self.plant_detector.apply_otsu(index_masked)
        
        # Дополнительно применяем маску к бинарному изображению
        binary = cv2.bitwise_and(binary, binary, mask=crop_mask)
        
        # Морфологическая обработка
        bitmap = self.plant_detector.morph_processing(binary)
        
        # Детекция контуров
        s_cnt, m_cnt, l_cnt = self.plant_detector.detect_plants(
            bitmap, 
            original_rgb.shape[:2]
        )
        
        # Масштабирование если было уменьшение
        if self.plant_detector.downscale_factor < 1.0:
            scale = 1.0 / self.plant_detector.downscale_factor
            s_cnt = [(cnt * scale).astype(np.int32) for cnt in s_cnt]
            m_cnt = [(cnt * scale).astype(np.int32) for cnt in m_cnt]
            l_cnt = [(cnt * scale).astype(np.int32) for cnt in l_cnt]
        
        data['crop_contours'] = {
            'small': s_cnt,
            'medium': m_cnt,
            'large': l_cnt
        }
        
        # Создаём изображение с bounding box'ами культур
        bboxes_img = original_rgb.copy()
        thickness = VISUALIZATION_CONFIG['bbox_thickness']
        colors = [
            VISUALIZATION_CONFIG['crop_small_color'],
            VISUALIZATION_CONFIG['crop_medium_color'],
            VISUALIZATION_CONFIG['crop_large_color']
        ]
        
        total_count = 0
        for contours, color in zip([s_cnt, m_cnt, l_cnt], colors):
            for cnt in contours:
                if isinstance(cnt, np.ndarray) and cnt.size > 0 and cnt.shape[0] > 0:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(bboxes_img, (x, y), (x + w, y + h), color, thickness)
                    total_count += 1
        
        data['crop_bboxes_img'] = bboxes_img
        data['crop_count'] = total_count
        
        return data


class DrawingStep(ProcessingStep):
    """Этап 4: Объединение результатов и финальная отрисовка"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        frame = data['frame']
        weed_detections = data.get('weed_detections', [])
        crop_contours = data.get('crop_contours', {'small': [], 'medium': [], 'large': []})
        crop_count = data.get('crop_count', 0)
        weed_count = len(weed_detections)
        
        # Создаём итоговое изображение
        result = frame.copy()
        
        # Рисуем сорняки (жёлтым)
        weed_color = VISUALIZATION_CONFIG['weed_color']
        thickness = VISUALIZATION_CONFIG['bbox_thickness']
        font_scale = VISUALIZATION_CONFIG['font_scale']
        font_thickness = VISUALIZATION_CONFIG['font_thickness']
        
        for det in weed_detections:
            x, y, w, h = det['bbox']
            
            # Bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), weed_color, thickness)
            
            # Подпись "W"
            label = f"W"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Фон для текста
            cv2.rectangle(
                result,
                (x, y - text_h - baseline - 5),
                (x + text_w + 5, y + baseline),
                weed_color,
                -1
            )
            
            # Текст
            cv2.putText(
                result,
                label,
                (x + 2, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness
            )
        
        # Рисуем культуры (разными цветами по размерам)
        crop_colors = [
            VISUALIZATION_CONFIG['crop_small_color'],
            VISUALIZATION_CONFIG['crop_medium_color'],
            VISUALIZATION_CONFIG['crop_large_color']
        ]
        size_labels = ['S', 'M', 'L']
        
        for contours, color, label in zip(
            [crop_contours['small'], crop_contours['medium'], crop_contours['large']],
            crop_colors,
            size_labels
        ):
            for cnt in contours:
                if isinstance(cnt, np.ndarray) and cnt.size > 0 and cnt.shape[0] > 0:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Bounding box
                    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
                    
                    # Подпись размера
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                    )
                    
                    # Фон для текста
                    cv2.rectangle(
                        result,
                        (x, y - text_h - baseline - 5),
                        (x + text_w + 5, y + baseline),
                        color,
                        -1
                    )
                    
                    # Текст
                    cv2.putText(
                        result,
                        label,
                        (x + 2, y - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        font_thickness
                    )
        
        # Добавляем счётчики
        counter_pos = VISUALIZATION_CONFIG['counter_position']
        counter_font_scale = VISUALIZATION_CONFIG['counter_font_scale']
        counter_font_thickness = VISUALIZATION_CONFIG['counter_font_thickness']
        counter_color = VISUALIZATION_CONFIG['counter_color']
        shadow_color = VISUALIZATION_CONFIG['counter_shadow_color']
        
        # Тень
        cv2.putText(
            result,
            f"Weeds: {weed_count} | Crops: {crop_count}",
            (counter_pos[0] + 2, counter_pos[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            counter_font_scale,
            shadow_color,
            counter_font_thickness + 2
        )
        
        # Основной текст
        cv2.putText(
            result,
            f"Weeds: {weed_count} | Crops: {crop_count}",
            counter_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            counter_font_scale,
            counter_color,
            counter_font_thickness
        )
        
        data['result_frame'] = result
        data['weed_count'] = weed_count
        data['crop_count'] = crop_count
        
        return data


class GreenOnGreenPipeline:
    """
    Конвейер для режима Green-on-Green
    Управляет последовательностью этапов обработки
    """
    
    def __init__(self, weed_detector: WeedDetector, plant_detector: PlantDetector):
        """
        Инициализация конвейера
        
        Args:
            weed_detector: Детектор сорняков (YOLO)
            plant_detector: Детектор культур (вегетационные индексы)
        """
        self.steps = [
            WeedDetectionStep(weed_detector),
            MaskingStep(),
            CropDetectionStep(plant_detector),
            DrawingStep()
        ]
    
    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Запуск конвейера обработки кадра
        
        Args:
            frame: Входной кадр (BGR)
            
        Returns:
            Словарь с результатами:
                - result_frame: Кадр с отрисованными детекциями
                - weed_count: Количество сорняков
                - crop_count: Количество культур
                - weed_detections: Список детекций сорняков
                - crop_contours: Контуры культур по размерам
        """
        data = {
            'frame': frame,
            'weed_detections': [],
            'weed_mask': None,
            'crop_mask': None,
            'crop_contours': {'small': [], 'medium': [], 'large': []},
            'crop_count': 0,
            'weed_count': 0
        }
        
        # Последовательное выполнение этапов
        for step in self.steps:
            data = step.process(data)
        
        return data


class GreenOnGreenDetector:
    """
    Главный класс режима Green-on-Green
    Предоставляет простой интерфейс для использования конвейера
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к модели YOLO (опционально)
        """
        # Инициализация детекторов
        self.weed_detector = WeedDetector(model_path=model_path)
        self.plant_detector = PlantDetector(
            index_type=PLANT_DETECTOR_CONFIG['index_type'],
            downscale_factor=PLANT_DETECTOR_CONFIG['downscale_factor']
        )
        
        # Создание конвейера
        self.pipeline = GreenOnGreenPipeline(
            self.weed_detector,
            self.plant_detector
        )
        
        print("✅ GreenOnGreenDetector initialized")
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Обработка кадра в режиме Green-on-Green
        
        Args:
            frame: Входной кадр (BGR)
            
        Returns:
            Tuple containing:
                - result_frame: Кадр с отрисованными сорняками (жёлтый) и культурами (зелёный/синий/красный)
                - weed_count: Количество обнаруженных сорняков
                - crop_count: Количество обнаруженных культур
        """
        if frame is None or frame.size == 0:
            return frame, 0, 0
        
        results = self.pipeline.process(frame)
        
        return (
            results['result_frame'],
            results['weed_count'],
            results['crop_count']
        )
    
    def get_detailed_results(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Получение подробных результатов обработки
        
        Args:
            frame: Входной кадр
            
        Returns:
            Полный словарь результатов от конвейера
        """
        return self.pipeline.process(frame)
