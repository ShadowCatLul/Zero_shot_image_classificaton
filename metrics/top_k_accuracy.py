from typing import List, Any, Union
import numpy as np
from .base_metrics import BaseMetric

class TopKAccuracyMetric(BaseMetric):
    """Класс для расчета Top-K Accuracy метрики"""
    
    def __init__(self, k: int = 1):
        """
        Args:
            k: Количество топовых предсказаний для учета
        """
        self.k = k
    
    def compute(self, predictions: List[List[Any]], targets: List[Any]) -> float:
        """Вычисление Top-K Accuracy
        
        Args:
            predictions: Список списков предсказаний для каждого примера
            targets: Истинные значения
            
        Returns:
            float: Значение Top-K Accuracy
        """
        correct = 0
        total = len(targets)
        
        for pred_list, target in zip(predictions, targets):
            if target in pred_list[:self.k]:
                correct += 1
                
        return correct / total if total > 0 else 0.0
    
    def get_name(self) -> str:
        return f"Top{self.k}Accuracy" 