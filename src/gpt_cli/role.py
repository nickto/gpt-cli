from enum import Enum


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
