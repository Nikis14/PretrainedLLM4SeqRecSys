from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    @abstractmethod
    def get_embeddings(self, description: List[str]) -> List[List[float]]:
        pass
