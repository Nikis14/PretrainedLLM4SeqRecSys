from time import sleep
from typing import Optional, List, Any
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.llm.llm import LLM
from src.llm.embedder import Embedder


class OpenAILLM(LLM):
    def __init__(self, token: str, model="gpt-4o-mini", max_tokens: int = None):
        """
        Initializes the OpenAIAdapter with the provided API token and model.

        Args:
            token (str): The API token for authenticating with the OpenAI API.
            model (str, optional): The model to use for generating completions. Defaults to "gpt-3.5-turbo".
        """
        self.name = "openai"
        self.token = token
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.llm = ChatOpenAI(
            openai_api_key=self.token,
            model=self.model,
            temperature=0.7,
            max_tokens=max_tokens,
        )

    def get_description(
        self, messages: List[List[Any]], max_output_tokens: Optional[int]
    ):
        if not isinstance(messages[0], list):
            messages = [messages]
        result = self.llm.batch(messages, config={"max_concurrency": 8})
        return [res.content for res in result]

    def prepare_input(self, prompt: str) -> List[Any]:
        return [HumanMessage(content=prompt)]

    def get_max_context_length(self):
        return 128000

    def get_tokens_length(self, messages: List[str] | str) -> int:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return len(self.tokenizer.encode(str(messages)))


class OpenAIEmbedder(Embedder): ...
