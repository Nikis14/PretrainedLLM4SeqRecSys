from abc import ABC, abstractmethod
from typing import List, Optional, Any


class LLM(ABC):
    name: str = None

    @abstractmethod
    def get_description(
        self, messages: List[List[str]] | List[str], max_output_tokens: Optional[int]
    ) -> str | List[str]:
        pass

    def get_max_context_length(self) -> int:
        pass

    def get_tokens_length(self, messages: List[str] | str) -> int:
        pass

    def prepare_input(self, prompt: str) -> Any:
        pass
