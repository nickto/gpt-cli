from __future__ import annotations

from typing import IO, Dict, List

import tiktoken

from .role import Role


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

    def __str__(self):
        return f"{self.role.value.title()}: {self.content}"


class History:
    messages: List[Message]
    system: Message

    def __init__(self, model: str):
        self.model = model
        self.system = Message(role=Role.system, content=None, model=self.model)
        self.messages = []

    def _add_message(self, message: Message) -> History:
        self.messages.append(message)
        return self

    def add_message(self, content: str, role: Role, n_tokens: int | None = None):
        match role:
            case Role.system:
                self.add_system(content=content, n_tokens=n_tokens)
            case Role.user:
                self.add_user(content=content, n_tokens=n_tokens)
            case Role.assistant:
                self.add_assistant(content=content, n_tokens=n_tokens)
            case _:
                raise ValueError("Unexpected value of role.")

    def add_user(self, content: str, n_tokens: int | None = None) -> History:
        message = Message(
            role=Role.user, content=content, model=self.model, n_tokens=n_tokens
        )
        self._add_message(message)
        return self

    def add_assistant(self, content: str, n_tokens: int | None = None) -> History:
        message = Message(
            role=Role.assistant, content=content, model=self.model, n_tokens=n_tokens
        )
        self._add_message(message)
        return self

    def add_system(self, content: str, n_tokens: int | None = None) -> History:
        message = Message(
            role=Role.system, content=content, model=self.model, n_tokens=n_tokens
        )
        self.system = message
        return self

    def is_system_set(self) -> bool:
        return self.system.content is not None

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

        if self.is_system_set():
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

    @staticmethod
    def _is_system_line(text: str) -> bool:
        return text == "System:\n"

    @staticmethod
    def _is_user_line(text: str) -> bool:
        return text == "User:\n"

    @staticmethod
    def _is_assistant_line(text: str) -> bool:
        return text == "Assistant:\n"

    def load(self, file: IO) -> History:
        self.messages: List[Message] = []
        content: str | None = None
        role: Role | None = None
        for line in file.readlines():
            if self._is_system_line(line):
                # If accumulated content, save it to message
                if content:
                    content = content.strip()
                    self.add_message(content=content, role=role)
                # Start accumulating new content
                role = Role.system
                content = ""
            elif self._is_user_line(line):
                # If accumulated content, save it to message
                if content:
                    content = content.strip()
                    self.add_message(content=content, role=role)
                # Start accumulating new content
                role = Role.user
                content = ""
            elif self._is_assistant_line(line):
                # If accumulated content, save it to message
                if content:
                    content = content.strip()
                    self.add_message(content=content, role=role)
                # Start accumulating new content
                role = Role.assistant
                content = ""
            elif content is not None:
                content += line
            else:
                # `content` is None, which means that the history file was improperly
                # formatted
                raise ValueError("History file is improperly formatted.")
        content = content.strip()
        self.add_message(content=content, role=role)
        return self
