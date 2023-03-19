from typing import List

import openai
from openai import Completion

from .key import OpenaiApiKey


class Instruct:
    def __init__(
        self,
        api_key: OpenaiApiKey,
        model: str = "gpt-3.5-turbo",
        out: str = None,
        stop: List[str] | None = None,
        max_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        openai.api_key = api_key.get()
        self.model = model
        self.out = out

        super().__init__(
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

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

        if self.out:
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
