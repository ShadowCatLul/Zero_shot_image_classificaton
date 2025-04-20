from typing import List, Any
import numpy as np
from sklearn.metrics import f1_score
from .base_metrics import BaseMetric

class F1Metric(BaseMetric):
    """Класс для расчета F1 метрики"""
    
    def __init__(self, average: str = 'weighted'):
        """
        Args:
            average: Способ усреднения ('micro', 'macro', 'weighted', 'samples')
        """
        self.average = average
    
    def compute(self, predictions: List[Any], targets: List[Any]) -> float:
        """Вычисление F1 метрики
        
        Args:
            predictions: Предсказания модели
            targets: Истинные значения
            
        Returns:
            float: Значение F1 метрики
        """
        return f1_score(targets, predictions, average=self.average)
    
    def get_name(self) -> str:
        return f"F1_{self.average}" 