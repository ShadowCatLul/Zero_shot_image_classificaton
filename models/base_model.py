from abc import ABC, abstractmethod
from typing import List, Any, Dict

class BaseModel(ABC):
    """Базовый класс для всех моделей"""
    
    @abstractmethod
    def predict(self, inputs: List[Any]) -> List[Any]:
        """Получение предсказаний модели
        
        Args:
            inputs: Входные данные
            
        Returns:
            List[Any]: Предсказания модели
        """
        pass
    
    @abstractmethod
    def predict_proba(self, inputs: List[Any]) -> List[List[float]]:
        """Получение вероятностей предсказаний
        
        Args:
            inputs: Входные данные
            
        Returns:
            List[List[float]]: Вероятности для каждого класса
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Получение названия модели"""
        pass 