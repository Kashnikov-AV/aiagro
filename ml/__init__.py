"""
ML модуль для AIagro
Содержит детекторы на основе машинного обучения
"""
from .weed_detector import WeedDetector
from .green_on_green_detector import GreenOnGreenDetector

__all__ = ['WeedDetector', 'GreenOnGreenDetector']