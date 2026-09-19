"""
Factory для создания детекторов по режиму работы
Реализует паттерн Factory для упрощения переключения режимов
"""
from typing import Optional, Any
from core.plant_detector import PlantDetector


class DetectorFactory:
    """
    Фабрика детекторов
    
    Создаёт экземпляры детекторов в зависимости от режима работы.
    Избавляет главный интерфейс от условных конструкций if/else.
    """
    
    _detectors = {}  # Кэш созданных детекторов
    
    @classmethod
    def create(cls, mode: str, model_path: Optional[str] = None) -> Any:
        """
        Создание детектора по режиму
        
        Args:
            mode: Режим работы ('standard', 'green_on_green')
            model_path: Путь к модели YOLO (для режима green_on_green)
            
        Returns:
            Экземпляр детектора для указанного режима
            
        Raises:
            ValueError: Если режим не поддерживается
            ImportError: Если отсутствуют необходимые зависимости
        """
        if mode == 'standard':
            return cls._create_standard_detector()
        
        elif mode == 'green_on_green':
            return cls._create_green_on_green_detector(model_path)
        
        else:
            raise ValueError(
                f"Unknown processing mode: {mode}. "
                f"Available modes: standard, green_on_green"
            )
    
    @classmethod
    def _create_standard_detector(cls) -> PlantDetector:
        """Создание стандартного детектора растений"""
        from config.settings import PLANT_DETECTOR_CONFIG
        
        # Проверяем кэш
        if 'standard' not in cls._detectors:
            detector = PlantDetector(
                index_type=PLANT_DETECTOR_CONFIG['index_type'],
                downscale_factor=PLANT_DETECTOR_CONFIG['downscale_factor']
            )
            cls._detectors['standard'] = detector
        
        return cls._detectors['standard']
    
    @classmethod
    def _create_green_on_green_detector(cls, model_path: Optional[str] = None):
        """Создание детектора для режима Green-on-Green"""
        from ml.green_on_green_detector import GreenOnGreenDetector
        
        # Для этого режима не используем кэш, так как может потребоваться разная конфигурация
        detector = GreenOnGreenDetector(model_path=model_path)
        return detector
    
    @classmethod
    def clear_cache(cls):
        """Очистка кэша детекторов"""
        cls._detectors.clear()
    
    @classmethod
    def get_available_modes(cls) -> list:
        """Возвращает список доступных режимов"""
        return ['standard', 'green_on_green']
