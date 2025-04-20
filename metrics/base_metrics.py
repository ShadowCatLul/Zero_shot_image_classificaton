from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Union

class BaseMetric(ABC):
    """Базовый класс для всех метрик"""
    
    @abstractmethod
    def compute(self, predictions: List[Any], targets: List[Any]) -> float:
        """Вычисление метрики
        
        Args:
            predictions: Предсказания модели
            targets: Истинные значения
            
        Returns:
            float: Значение метрики
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Получение названия метрики"""
        pass 