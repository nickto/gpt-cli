from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from .constants import CONFIG_DIR

_kb = KeyBindings()


@_kb.add(Keys.Enter)
def _(event):
    if event.current_buffer.text.endswith("\\"):
        event.current_buffer.newline()
    else:
        event.current_buffer.validate_and_handle()


@_kb.add(Keys.Escape, Keys.Enter)
def _(event):
    event.current_buffer.newline()


def prompt() -> str:
    path = Path(CONFIG_DIR) / Path(".history")
    session = PromptSession(
        history=FileHistory(str(path)),
        key_bindings=_kb,
    )
    user_input = session.prompt("> ", prompt_continuation="  ")

    # Clean up user input
    user_input = user_input.split("\n")
    # Remove  whitespace
    user_input = [line.strip() for line in user_input]
    # Remove trailing slahes
    user_input = map(
        lambda line: line if (len(line) > 0 and line[-1] != "\\") else line[:-1],
        user_input,
    )
    # Concat back to string
    user_input = "\n".join(user_input)

    return user_input
