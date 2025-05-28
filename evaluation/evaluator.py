from typing import List, Dict, Any
from models.base_model import BaseModel
from metrics.base_metrics import BaseMetric

class ModelEvaluator:
    """Класс для оценки моделей"""
    
    def __init__(self, metrics: List[BaseMetric]):
        """
        Args:
            metrics: Список метрик для оценки
        """
        self.metrics = metrics
    
    def evaluate(self, model: BaseModel, inputs: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Оценка модели на заданных данных
        
        Args:
            model: Модель для оценки
            inputs: Входные данные
            targets: Истинные значения
            
        Returns:
            Dict[str, float]: Словарь с результатами метрик
        """
        predictions = model.predict(inputs)
        results = {}
        
        for metric in self.metrics:
            metric_value = metric.compute(predictions, targets)
            results[metric.get_name()] = metric_value
            
        return results
    
    def evaluate_with_proba(self, model: BaseModel, inputs: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Оценка модели с использованием вероятностей
        
        Args:
            model: Модель для оценки
            inputs: Входные данные
            targets: Истинные значения
            
        Returns:
            Dict[str, float]: Словарь с результатами метрик
        """
        probabilities = model.predict_proba(inputs)
        results = {}
        
        for metric in self.metrics:
            if hasattr(metric, 'compute_with_proba'):
                metric_value = metric.compute_with_proba(probabilities, targets)
                results[metric.get_name()] = metric_value
                
        return results 