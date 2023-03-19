from abc import ABC, abstractmethod
from typing import Dict, List

import rich
import typer
from rich.prompt import Prompt


class AbstractCompletion(ABC):
    completion_params: Dict[str, str | float | int | List[str]]

    def __init__(self, **completion_params):
        # Check and fix some model params if needed
        if "stop" in completion_params:
            stop = completion_params["stop"]
            stop = stop if stop else None  # "" or [] becomes None
            completion_params["stop"] = stop
        if "max_tokens" in completion_params:
            assert completion_params["max_tokens"] > 0
        if "temperature" in completion_params:
            assert 0 <= completion_params["temperature"] <= 2
        if "top_p" in completion_params:
            assert 0 <= completion_params["top_p"] <= 1
        if "n" in completion_params:
            assert completion_params["n"] > 0
        if "presence_penalty" in completion_params:
            assert -2 <= completion_params["presence_penalty"] <= 2
        if "frequency_penalty" in completion_params:
            assert -2 <= completion_params["frequency_penalty"] <= 2
        self.completion_params = completion_params

    def ask_for_input(self) -> str:
        prompt = Prompt
        prompt.prompt_suffix = "> "
        user_input = prompt.ask()
        if user_input in ("exit", "quit", ":q"):
            raise typer.Exit()
        rich.print()

        return user_input


class AbstractChat(AbstractCompletion, ABC):
    @abstractmethod
    def start(self):
        raise NotImplementedError


class AbstractPrompt(AbstractCompletion, ABC):
    @abstractmethod
    def prompt(self, user_input: str) -> List[str]:
        raise NotImplementedError
