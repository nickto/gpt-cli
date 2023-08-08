from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import openai
import rich
import typer
from openai import ChatCompletion
from openai.error import APIError, RateLimitError, ServiceUnavailableError
from rich.markdown import Markdown
from rich.prompt import Prompt

from chatgpt_cli import pretty

from .history import History
from .key import OpenaiApiKey
from .role import Role


class AbstractChat(ABC):
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

    @staticmethod
    def ask_for_input() -> str:
        prompt = Prompt
        prompt.prompt_suffix = "> "
        user_input = ""
        while True:
            user_input += prompt.ask()
            if user_input.strip()[-1] != "\\":
                break
            else:
                # Change \ for \n
                user_input = user_input[:-1]
                user_input += "\n"
        if user_input in ("exit", "quit", ":q"):
            raise typer.Exit()

        return user_input

    @abstractmethod
    def start(self):
        raise NotImplementedError


class Chat(AbstractChat):
    RETRY_SLEEP: int = 10

    def __init__(
        self,
        api_key: OpenaiApiKey,
        model: str = "gpt-3.5-turbo",
        system: str | None = None,
        out: str = None,
        history: History = None,
        stop: List[str] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        openai.api_key = api_key.get()
        self.system = system
        self.model = model
        self.out = out

        if history:
            self.history = history
        else:
            self.history = History(model=model)

        # Set system only if not present in history. This could happen if history
        # is loaded from a file.
        if system and not self.history.is_system_set():
            self.history.add_system(system)

        super().__init__(
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

    def _get_max_model_tokens(self) -> int:
        if "16k" in self.model:
            max_tokens = 2**14 // 2  # half of 16K
        elif "32k" in self.model:
            max_tokens = 2**15 // 2  # half of 32K
        elif "gpt-4" in self.model:
            max_tokens = 2**13 // 2  # half of 8K, which is gpt-4 default
        else:
            # This covers, among others, "gpt-3.5"
            max_tokens = 2**12 // 2  # half of 4K, which is gpt-3.5 default
        return max_tokens

    def _need_user_input(self) -> bool:
        if len(self.history.messages) == 0:
            # First message should be user input
            return True
        # Need input, if last message was not user message
        last_message = self.history.messages[-1]
        return last_message.role != Role.user

    def start(self):
        if self.completion_params["n"] > 1:
            msg = "Cannot use more than 1 completion in interactive chat mode."
            raise ValueError(msg)

        if self.out and self.system:
            with open(self.out, "w+") as f:
                f.write("System:\n" + self.system)
                f.write("\n\n")

                for message in self.history.messages:
                    if message.role == Role.user:
                        f.write("User:\n" + message.content)
                    elif message.role == Role.assistant:
                        f.write("Assistant:\n" + message.content)
                    f.write("\n\n")

        while True:
            if self._need_user_input():
                user_input = self.ask_for_input()
                self.history.add_user(content=user_input)

                if self.out is not None:
                    with open(self.out, "a+") as f:
                        f.write("User:\n" + user_input)
                        f.write("\n\n")

            success = False
            while not success:
                try:
                    completion = pretty.typing_animation(
                        ChatCompletion.create,
                        model=self.model,
                        messages=self.history.get_messages(
                            max_tokens=self._get_max_model_tokens()
                        ),
                        **self.completion_params,
                    )
                    success = True
                except RateLimitError:
                    msg = f"RateLimitError: retrying in {self.RETRY_SLEEP:d} seconds."
                    pretty.waiting_animation(self.RETRY_SLEEP, msg)
                except APIError:
                    msg = f"APIError: retrying in {self.RETRY_SLEEP:d} seconds."
                    pretty.waiting_animation(self.RETRY_SLEEP, msg)
                except ServiceUnavailableError:
                    msg = f"ServiceUnavailableError: retrying in {self.RETRY_SLEEP:d} seconds."
                    pretty.waiting_animation(self.RETRY_SLEEP, msg)

            assistant_reply = completion.choices[0].message["content"]
            self.history.add_assistant(
                content=assistant_reply,
                n_tokens=completion["usage"]["completion_tokens"],
            )
            rich.print()
            rich.print(Markdown(assistant_reply))
            rich.print()

            if self.out:
                with open(self.out, "a+") as f:
                    f.write("Assistant:\n" + assistant_reply)
                    f.write("\n\n")
