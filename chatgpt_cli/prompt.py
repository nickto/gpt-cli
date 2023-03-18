from .config import Config
from typing import List
import openai
from openai import Completion


class Prompt:
    def __init__(
        self,
        config: Config,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        openai.api_key = config.get_api_key()
        self.model = model
        self.config = config

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def prompt(self, user_input: str) -> List[str]:
        response = Completion.create(
            model=self.model,
            prompt=user_input,
            temperature=self.temperature,
            n=self.n,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )
        completions = [c["text"] for c in response.choices]
        return completions
