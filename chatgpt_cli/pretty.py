from rich import print
import typer
import openai
import typer
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Callable


def error(text: str) -> None:
    print(f"[bold red]Error:[/bold red] {text}")


def warning(text: str) -> None:
    print(f"[bold yellow]Warning:[/bold yellow] {text}")


def typing_animation(func: Callable, *args, **kwargs):
    with Progress(
        SpinnerColumn(),
        TextColumn("Typing..."),
        transient=True,
    ) as progress:
        progress.add_task(description="Processing request", total=None)
        response = func(*args, **kwargs)
    return response


if __name__ == "__main__":
    "Example usage."
    import time

    warning("Test warning")
    error("Test error")
    typing_animation(lambda _: time.sleep(1), None)
