from typing import List, Any, Optional
from src.llm.llm import LLM
from src.llm.embedder import Embedder
from time import sleep
from gigachat.exceptions import ResponseError
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage


class GigaChatLLM(LLM):
    def __init__(self, token: str, max_tokens: int = None):
        """
        Initializes the GigaChatAdapter with the provided API token and model.

        Args:
            token (str): The API token for authenticating with the GigaChatAdapter API.
        """
        self.name = "gigachat"
        self.token = token
        self.model_name = "GigaChat-Pro"
        self.llm = GigaChat(
            credentials=self.token,
            scope="GIGACHAT_API_CORP",
            model=self.model_name,
            verify_ssl_certs=False,
            max_tokens=max_tokens,
        )

    def get_description(
        self, messages: List[List[Any]], max_output_tokens: Optional[int]
    ):
        if not isinstance(messages[0], list):
            messages = [messages]
        try:
            result = self.llm.batch(
                messages,
                config={
                    "max_concurrency": 2,
                    "timeout": 60,
                    "max_output_tokens": max_output_tokens,
                },
            )
            return [res.content for res in result]
        except ResponseError:
            debug_messages = [HumanMessage(content="Test")]
            self.llm.batch([debug_messages] * 10, config={"max_concurrency": 4})
            result = self.llm.batch(
                messages,
                config={
                    "max_concurrency": 2,
                    "timeout": 60,
                    "max_output_tokens": max_output_tokens,
                },
            )
            return [res.content for res in result]

    def prepare_input(self, prompt: str) -> List[Any]:
        return [HumanMessage(content=prompt)]

    def get_max_context_length(self):
        return 32000

    def get_tokens_length(self, messages: List[str] | str) -> int:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        try:
            return self.llm.tokens_count(input_=[str(messages)], model=self.model_name)[
                0
            ].tokens
        except ResponseError:
            return len(str(messages)) // 3


class GigaChatEmbedder(Embedder):
    def __init__(self, token: str):
        """
        Initializes the GigaChatAdapter with the provided API token and model.

        Args:
            token (str): The API token for authenticating with the GigaChat API.
        """
        self.token = token
        self.embedder = GigaChat(credentials=self.token)

    def get_embeddings(self, text: str):
        """
        Gets the embedding for the provided text.

        Args:
            text (str): The text to get the embedding for.

        Returns:
            list: The embedding of the provided text.
        """
        return self.embedder.embeddings(texts=[text])
