from .config import Config
from typing import List
import openai
from openai import Completion


class Instruct:
    def __init__(
        self,
        config: Config,
        out: str = None,
        model: str = "gpt-3.5-turbo",
        stop: List[str] | None = None,
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
        self.out = out

        self.stop = stop if stop else None
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
            stop=self.stop,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            temperature=self.temperature,
            n=self.n,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )
        completions = [c["text"] for c in response.choices]

        if self.out is not None:
            with open(self.out, "w+") as f:
                f.write("User:\n")
                f.write(user_input + "\n\n")
                for i, completion in enumerate(completions):
                    if len(completions) > 1:
                        header = f"Assistant, completion {i}/{len(completions):d}:"
                    else:
                        header = "Assistant:"
                    f.write(header + "\n")
                    f.write(completion)

        return completions
