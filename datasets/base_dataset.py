from abc import ABC, abstractmethod
from typing import List, Any, Tuple

class BaseDataset(ABC):
    @abstractmethod
    def get_samples(self, n_samples: int = None) -> Tuple[List[Any], List[int]]:
        """Возвращает пути к изображениям и метки"""
        pass

    @abstractmethod
    def get_classnames(self) -> List[str]:
        """Возвращает список названий классов"""
        pass