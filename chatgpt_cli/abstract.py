from abc import ABC, abstractmethod
from typing import List
import rich
import typer
from rich.prompt import Prompt


class AbstractCompletion(ABC):
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
