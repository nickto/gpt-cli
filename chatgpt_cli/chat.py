from __future__ import annotations

from enum import Enum
from pprint import pprint
from typing import Dict, List

import openai
import tiktoken
from openai import ChatCompletion
import rich
from rich.markdown import Markdown
from rich.prompt import Prompt

from chatgpt_cli import pretty
from .config import Config


class Chat:
    def __init__(
        self,
        config: Config,
        out: str = None,
        system: str | None = None,
        model: str = "gpt-3.5-turbo",
        stop: List[str] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 1,
        n: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        openai.api_key = config.get_api_key()
        self.config = config
        self.out = out
        self.system = system

        self.history = History(model=model)
        if system is not None:
            self.history.add_system(system)
        self.model = model

        self.stop = stop if stop else None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def start(self):
        if self.n > 1:
            msg = "cannot return more than 1 completion in a chat mode, hence will return only 1."
            pretty.warning(msg)

        if self.out is not None:
            with open(self.out, "w+") as f:
                f.write("System:\n" + self.system)
                f.write("\n\n")

        while True:
            prompt = Prompt
            prompt.prompt_suffix = "> "
            user_input = prompt.ask()
            rich.print()

            user_message = Message(
                role=Role.user,
                content=user_input,
                model=self.model,
            )
            self.history.add_message(user_message)

            if self.out is not None:
                with open(self.out, "a+") as f:
                    f.write("User:\n" + user_input)
                    f.write("\n\n")

            completion = pretty.typing_animation(
                ChatCompletion.create,
                model=self.model,
                messages=self.history.get_messages(),
                stop=self.stop,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                n=1,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
            )

            reply = completion.choices[0].message["content"]
            rich.print(Markdown(reply))
            rich.print()

            if self.out is not None:
                with open(self.out, "a+") as f:
                    f.write("Assistant:\n" + reply)
                    f.write("\n\n")

    def prompt(self, user_input: str) -> List[str]:
        user_message = Message(
            role=Role.user,
            content=user_input,
            model=self.model,
        )
        self.history.add_message(user_message)

        response = pretty.typing_animation(
            ChatCompletion.create,
            model=self.model,
            messages=self.history.get_messages(),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )

        completions = [c.message["content"] for c in response.choices]

        if self.out is not None:
            with open(self.out, "w+") as f:
                if self.system is not None:
                    f.write("User:\n")
                    f.write(self.system + "\n\n")

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


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message:
    def __init__(
        self,
        role: Role,
        content: str | None,
        n_tokens: int | None = None,
        model: str | None = None,
    ):
        self.role = role
        self.content = content

        # Content can be None only if role is "system". This is allowed to
        # distinguish between no system message provided and empty string
        # message provided for system, because model behaves differently in
        # these two cases.
        if content is None:
            if role == Role.system:
                n_tokens = 0
            else:
                raise ValueError('If role is not "system", content cannot be None')

        if n_tokens is None:
            if model is None:
                raise ValueError("If n_tokens not present, model should be present.")
            self.n_tokens = self.count_tokens(content, model)
        else:
            self.n_tokens = n_tokens

    @staticmethod
    def count_tokens(text: str, model: str) -> int:
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(text))
        return num_tokens


class History:
    messages: List[Message]

    def __init__(self, model: str):
        self.model = model
        self.system = Message(role=Role.system, content=None, model=self.model)
        self.messages = []

    def add_message(self, message: Message) -> History:
        self.messages.append(message)
        return self

    def add_user(self, content: str, n_tokens: int | None = None) -> History:
        message = Message(
            role=Role.user, content=content, model=self.model, n_tokens=n_tokens
        )
        self.add_message(message)
        return self

    def add_assistant(self, content: str, n_tokens: int | None = None) -> History:
        message = Message(
            role=Role.assistant, content=content, model=self.model, n_tokens=n_tokens
        )
        self.add_message(message)
        return self

    def add_system(self, content: str, n_tokens: int | None = None) -> History:
        message = Message(
            role=Role.system, content=content, model=self.model, n_tokens=n_tokens
        )
        self.system = message
        return self

    def get_history(
        self,
        max_tokens: int = 2048,
        max_messages: int = 32 * 1024,  # just a very large number
    ) -> List[Message]:
        history_tokens = self.system.n_tokens
        history = []
        for message in reversed(self.messages):
            # Will token count be ok?
            ok_to_add = (history_tokens + message.n_tokens) <= max_tokens
            # Will message count be ok?
            ok_to_add = ok_to_add and (len(history) < max_messages)

            if ok_to_add:
                history.insert(0, message)
                history_tokens += message.n_tokens
            else:
                break

        if self.system.content is not None:
            history.insert(0, self.system)

        return history

    @staticmethod
    def history2dict(history: List[Message]) -> Dict[str, str]:
        messages = []
        for message in history:
            messages.append({"role": message.role.value, "content": message.content})

        return messages

    def get_messages(
        self,
        max_tokens: int = 2048,
        max_messages: int = 32 * 1024,  # just a very large number
    ) -> Dict[str, str]:
        history = self.get_history(max_tokens=max_tokens, max_messages=max_messages)
        return self.history2dict(history)


if __name__ == "__main__":
    "Unit tests and example usage."
    # With system
    history = History(model="gpt-3.5-turbo")
    assert len(history.get_history()) == 0
    history.add_system("You are a helpful assistant.")
    assert len(history.get_history()) == 1
    history.add_user("Who is Banksy?")
    assert len(history.get_history()) == 2
    history.add_assistant("I don't know")
    assert len(history.get_history()) == 3
    pprint(history.get_messages())

    # Without system
    print()
    history = History(model="gpt-3.5-turbo")
    history.add_user("Who is Banksy?")
    assert len(history.get_history()) == 1
    history.add_assistant("I don't know")
    assert len(history.get_history()) == 2
    pprint(history.get_messages())

    # Limit messages
    print()
    history = History(model="gpt-3.5-turbo")
    history.add_system("You are a helpful assistant.")
    assert len(history.get_history()) == 1
    history.add_user("Who is Banksy?")
    assert len(history.get_history()) == 2
    history.add_assistant("I don't know")
    assert len(history.get_history()) == 3
    history.add_user("Are you sure?")
    assert len(history.get_history()) == 4
    history.add_assistant("Yes I am sure.")
    assert len(history.get_history(max_messages=2)) == 3  # 3 because have system
    pprint(history.get_messages(max_messages=2))

    # Limit tokens
    print()
    history = History(model="gpt-3.5-turbo")
    history.add_system("You are a helpful assistant.")
    assert len(history.get_history()) == 1
    history.add_user("Who is Banksy?")
    assert len(history.get_history()) == 2
    history.add_assistant("I don't know")
    assert len(history.get_history()) == 3
    history.add_user("Are you sure?")
    assert len(history.get_history()) == 4
    history.add_assistant("Yes I am sure.")
    assert len(history.get_history(max_tokens=15)) == 3  # 3 because have system
    pprint(history.get_messages(max_tokens=15))
