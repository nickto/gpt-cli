from __future__ import annotations
from .constants import CONFIG_DIR, OPENAI_API_KEY_FILENAME
import typer
from gpt_cli import pretty
import os


class OpenaiApiKey:
    openai_api_key: str
    path: str = os.path.join(CONFIG_DIR, OPENAI_API_KEY_FILENAME)

    def __init__(self, openai_api_key: str | None = None):
        if openai_api_key:
            self.openai_api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        else:
            if not os.path.exists(self.path):
                pretty.error(
                    f"{self.path} does not exist: could not read OpenAI API key. "
                    "Have you already run `gpt-cli init`?"
                )
                raise typer.Abort()
            with open(self.path, "r") as f:
                self.openai_api_key = f.read().strip()

    def get(self) -> str:
        return self.openai_api_key

    def set(self, openai_api_key: str) -> OpenaiApiKey:
        self.openai_api_key = openai_api_key
        return self

    def save(self) -> OpenaiApiKey:
        if self.openai_api_key is None:
            raise ValueError(
                "openai_api_key should not be None when attempting to save."
            )
        with open(self.path, "w+") as f:
            f.write(self.openai_api_key)
        os.chmod(self.path, 0o600)  # .rw-------
        return self
