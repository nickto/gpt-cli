import time
from typing import Callable

from rich import print
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Column


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


def waiting_animation(seconds: int = 1, msg: str = ""):
    progress = Progress(
        SpinnerColumn(),
        TextColumn(msg),
        BarColumn(bar_width=None, table_column=Column(ratio=2)),
        expand=False,
        transient=True,
    )
    with progress:
        for _ in progress.track(range(seconds * 10)):
            time.sleep(0.1)


if __name__ == "__main__":
    "Example usage."
    import time

    warning("Test warning")
    error("Test error")
    typing_animation(lambda _: time.sleep(1), None)
