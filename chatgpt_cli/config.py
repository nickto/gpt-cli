from typing import IO
from chatgpt_cli import pretty
import typer
import yaml
import os


class Config:
    api_key: str

    def __init__(self, config: IO):
        self.config = yaml.safe_load(config)
        self.api_key = self._get_api_key()

    def get_api_key(self) -> str:
        return self.api_key

    def _get_api_key(self) -> str:
        if os.getenv("OPENAI_API_KEY"):
            return os.getenv("OPENAI_API_KEY")
        else:
            try:
                if self.config is None:
                    pretty.error("OpenAI API key not found.")
                    raise typer.Exit(code=1)
                return self.config["authentication"]["openai_api_key"]
            except KeyError:
                pretty.error("OpenAI API key not found.")
                raise typer.Exit(code=1)
