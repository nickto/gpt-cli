from .config import Config
from typing import List
import openai
from openai import Completion


class Prompt:
    def __init__(
        self,
        config: Config,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        n: int = 1,
        max_tokens: int = 128,
        top_p: float = 1,
    ):
        openai.api_key = config.get_api_key()
        self.model = model
        self.config = config

        self.temperature = temperature
        self.n = n
        self.max_tokens = max_tokens
        self.top_p = top_p

    def prompt(self, user_input: str) -> List[str]:
        response = Completion.create(
            model=self.model,
            prompt=user_input,
            temperature=self.temperature,
            n=self.n,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        completions = [c["text"] for c in response.choices]
        return completions
