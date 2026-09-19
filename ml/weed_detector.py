"""
WeedDetector - Детектор сорняков на базе YOLO
Использует модель YOLOv8 для обнаружения сорняков на изображениях
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ Warning: ultralytics not installed. Install with: pip install ultralytics")

from config.settings import (
    WEED_MODEL_PATH,
    YOLO_CONFIG,
    VISUALIZATION_CONFIG
)


class WeedDetector:
    """
    Детектор сорняков на основе YOLO
    
    Атрибуты:
        model: Загруженная модель YOLO
        confidence_threshold: Порог уверенности для детекции
        iou_threshold: IoU порог для NMS
        input_size: Размер входа модели
    """
    
    def __init__(self, model_path: Optional[str] = None, override_config: Optional[Dict] = None):
        """
        Инициализация детектора сорняков
        
        Args:
            model_path: Путь к файлу модели .pt (по умолчанию из settings)
            override_config: Словарь для переопределения настроек YOLO_CONFIG
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics library is required for WeedDetector. "
                "Install it with: pip install ultralytics"
            )
        
        self.model_path = model_path or WEED_MODEL_PATH
        self.config = {**YOLO_CONFIG, **(override_config or {})}
        
        self.confidence_threshold = self.config['confidence_threshold']
        self.iou_threshold = self.config['iou_threshold']
        self.input_size = self.config['input_size']
        self.device = self.config.get('device', 'cpu')
        
        # Проверка наличия модели
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please train the model using ml/train_weed_detector.ipynb"
            )
        
        # Загрузка модели
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"✅ WeedDetector initialized with model: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Детекция сорняков на кадре
        
        Args:
            frame: Входное изображение (BGR, OpenCV format)
            
        Returns:
            Tuple containing:
                - List of dicts with weed detections: [{'bbox': [x,y,w,h], 'confidence': float}, ...]
                - Binary mask where weeds are white (255) and background is black (0)
        """
        if frame is None or frame.size == 0:
            return [], np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Запуск инференса YOLO
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False
        )
        
        detections = []
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Получаем координаты bounding box
                bbox_xyxy = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                
                x1, y1, x2, y2 = map(int, bbox_xyxy)
                width = x2 - x1
                height = y2 - y1
                
                # Формируем detection dict
                detection = {
                    'bbox': [x1, y1, width, height],  # [x, y, w, h]
                    'bbox_xyxy': [x1, y1, x2, y2],    # [x1, y1, x2, y2]
                    'confidence': confidence,
                    'class_id': int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0
                }
                detections.append(detection)
                
                # Рисуем на маске (белый прямоугольник)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return detections, mask
    
    def create_inverted_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Создаёт инвертированную маску (культуры = белые, сорняки = чёрные)
        
        Args:
            mask: Бинарная маска с сорняками (белые)
            
        Returns:
            Инвертированная маска
        """
        return cv2.bitwise_not(mask)
    
    def apply_mask_to_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Применяет маску к кадру (обнуляет области сорняков)
        
        Args:
            frame: Исходный кадр
            mask: Маска где сорняки = 255 (белые)
            
        Returns:
            Кадр с замаскированными сорняками
        """
        # Создаём копию
        masked_frame = frame.copy()
        # Обнуляем области сорняков (где маска = 255)
        masked_frame[mask > 0] = 0
        return masked_frame
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Отрисовывает детекции сорняков на кадре
        
        Args:
            frame: Исходный кадр
            detections: Список детекций от метода detect()
            
        Returns:
            Кадр с отрисованными bounding box'ами
        """
        result = frame.copy()
        color = VISUALIZATION_CONFIG['weed_color']
        thickness = VISUALIZATION_CONFIG['bbox_thickness']
        font_scale = VISUALIZATION_CONFIG['font_scale']
        font_thickness = VISUALIZATION_CONFIG['font_thickness']
        
        for det in detections:
            x, y, w, h = det['bbox']
            
            # Рисуем bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            # Подпись "W" и уверенность
            label = f"W {det['confidence']:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Фон для текста
            cv2.rectangle(
                result,
                (x, y - text_height - baseline - 5),
                (x + text_width + 5, y + baseline),
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
                (0, 0, 0),  # Чёрный текст на цветном фоне
                font_thickness
            )
        
        return result
