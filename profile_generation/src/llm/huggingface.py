from typing import Any, List, Optional, Union
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from src.llm.embedder import Embedder
from src.llm.llm import LLM


class E5Embedder(Embedder):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.model = SentenceTransformer(model_name).to("cuda")

    def get_embeddings(self, description: List[str]) -> List[List[float]]:
        return self.model.encode(description).tolist()


class E5SmallEmbedder(Embedder):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.model = SentenceTransformer(model_name).to("cuda")

    def get_embeddings(self, description: List[str]) -> List[List[float]]:
        return self.model.encode(description).tolist()


class KaLMEmbedder(Embedder):
    def __init__(
        self, model_name: str = "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1"
    ):
        self.model = SentenceTransformer(model_name).to("cuda")

    def get_embeddings(self, description: List[str]) -> List[List[float]]:
        return self.model.encode(description).tolist()


class HFLLM(LLM):
    def __init__(self, token, model_name, name):
        login(token)
        self.name = name

        from vllm import LLM as vLLM

        self.model = vLLM(
            model=model_name, tokenizer=model_name, dtype="float16", device="cuda"
        )

    def prepare_input(self, prompt: str) -> Any:
        return [{"role": "system", "content": prompt}]

    def get_description(
        self, messages: List[str] | List[List[str]], max_output_tokens: Optional[int]
    ) -> str | List[str]:
        from vllm import SamplingParams

        if not isinstance(messages[0], list):
            messages = [messages]

        prompt = [str(message[0]) for message in messages]

        sampling_params = SamplingParams(temperature=0.7, max_tokens=max_output_tokens)

        results = self.model.generate(prompt, sampling_params=sampling_params)

        return [result.outputs[0].text.strip() for result in results]

    def get_max_context_length(self):
        return 8192

    def get_tokens_length(self, messages: List[str] | str) -> int:
        if not isinstance(messages[0], list):
            messages = [messages]

        tokenizer = self.model.get_tokenizer()
        return len(tokenizer.encode(str(messages)))


class Gemma2(HFLLM):
    def __init__(self, token: str):
        super().__init__(token, "UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3", "gemma2")


class Llama3(HFLLM):
    def __init__(self, token: str):
        super().__init__(token, "NousResearch/Hermes-3-Llama-3.1-8B", "llama3")

    def get_max_context_length(self):
        return 65536
