from __future__ import annotations

from typing import Dict, List

import openai
import rich
import typer
from openai import ChatCompletion
from openai.error import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)
from rich.markdown import Markdown
from .prompt import prompt

from gpt_cli import pretty

from .constants import DEFAULT_SYSTEM
from .context import Context
from .key import OpenaiApiKey
from .message import Message
from .role import Role


class Chat:
    RETRY_SLEEP: int = 10
    chat_completion_params: Dict[str, str | float | int | List[str]]

    def __init__(
        self,
        api_key: OpenaiApiKey,
        model: str = "gpt-3.5-turbo",
        system: str | None = None,
        out: str = None,
        context: Context = None,
        stop: List[str] | None = None,
        max_completion_tokens: int | None = None,
        max_context_tokens: int | None = None,
        temperature: float = 0.2,
        top_p: float = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        openai.api_key = api_key.get()
        self.model = model
        self.out = out

        if context:
            self.context = context
        else:
            self.context = Context(model=model)

        if system is None:
            self.system = DEFAULT_SYSTEM
        if not self.context.is_system_set():
            self.context.set_system(system)

        # Infer and check token counts
        if max_completion_tokens is not None and max_completion_tokens < 0:
            pretty.error("--max-completion-tokens should be a positive integer.")
            quit(1)
        if max_context_tokens is not None and max_context_tokens < 0:
            pretty.error("--max-context-tokens should be a positive integer.")
            quit(1)

        if max_completion_tokens:
            self.max_completion_tokens = max_completion_tokens
        else:
            self.max_completion_tokens = self.model.max_tokens // 2

        if max_context_tokens:
            self.max_context_tokens = max_context_tokens
        else:
            self.max_context_tokens = self.model.max_tokens - self.max_completion_tokens
            if self.max_context_tokens < 0:
                pretty.error(
                    "'--max-completion-tokens' cannot be larger than the "
                    "maximum number of tokens allowed by the "
                    f"'{self.model.name}' model: {self.model.max_tokens:,d}."
                )
                quit(1)

        if self.max_completion_tokens + self.max_context_tokens > self.model.max_tokens:
            pretty.error(
                "Sum of '--max-completion-tokens' and '--max-context-tokens' "
                "cannot be larger than the "
                "maximum number of tokens allowed by the "
                f"'{self.model.name}' model: {self.model.max_tokens:,d}."
            )
            quit(1)

        self.stop = stop if stop else None  # "" or [] becomes None
        self.temperature = temperature
        assert 0 <= self.temperature <= 2
        self.top_p = top_p
        assert 0 <= self.top_p <= 1
        self.presence_penalty = presence_penalty
        assert -2 <= self.presence_penalty <= 2
        self.frequency_penalty = frequency_penalty
        assert -2 <= self.frequency_penalty <= 2

        self.chat_completion_params = {
            "stop": self.stop,
            "max_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

    @staticmethod
    def ask_for_input() -> str:
        user_input = prompt()
        if user_input.lower().strip() in ("exit", "quit", ":q"):
            raise typer.Exit()

        return user_input

    def _need_user_input(self) -> bool:
        if len(self.context.messages) == 0:
            # First message should be user input
            return True
        # Need input, if last message was not user message
        last_message = self.context.messages[-1]
        return last_message.role != Role.user

    def start(self):
        while True:
            # Check if we need user input
            if self._need_user_input():
                user_input = self.ask_for_input()
                self.context.add_message(
                    Message(content=user_input, role=Role.user, model=self.model)
                )
                if self.out:
                    self.context.save(self.out)

            # Send request
            success = False
            while not success:
                try:
                    completion = pretty.typing_animation(
                        ChatCompletion.create,
                        model=self.model.name,
                        messages=self.context.get_messages(
                            max_tokens=self.model.max_tokens,
                        ),
                        **self.chat_completion_params,
                    )
                    success = True
                except RateLimitError:
                    s = self.RETRY_SLEEP
                    msg = f"RateLimitError: retrying in {s:d} seconds."
                    pretty.waiting_animation(s, msg)
                except APIError:
                    s = self.RETRY_SLEEP
                    msg = f"APIError: retrying in {s:d} seconds."
                    pretty.waiting_animation(s, msg)
                except ServiceUnavailableError:
                    s = self.RETRY_SLEEP
                    msg = f"ServiceUnavailableError: retrying in {s:d} seconds."
                    pretty.waiting_animation(s, msg)
                except APIConnectionError:
                    s = self.RETRY_SLEEP
                    msg = f"APIConnectionError: retrying in {s:d} seconds."
                    pretty.waiting_animation(s, msg)
                except AuthenticationError:
                    msg = (
                        "Incorrect API key provided. You can find your API key "
                        "at https://platform.openai.com/account/api-keys. "
                        "Then rerun the 'init' command or specify it using the "
                        "environment variable 'OPENAI_API_KEY', or the command line "
                        "option '--openai-api-key'."
                    )
                    pretty.error(msg)
                    quit(1)

            assistant_reply = completion.choices[0].message["content"]
            self.context.add_message(
                Message(content=assistant_reply, role=Role.assistant, model=self.model)
            )
            rich.print()
            rich.print(Markdown(assistant_reply))
            rich.print()

            if self.out:
                self.context.save(self.out)
