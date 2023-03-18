import sys
from typing import Dict, List, Optional, Tuple

import rich
import typer
from rich import print
from rich.markdown import Markdown

from chatgpt_cli import pretty
from chatgpt_cli.chat import Chat
from chatgpt_cli.config import Config
from chatgpt_cli.prompt import Prompt

app = typer.Typer(rich_markup_mode="markdown")


CONFIG_OPTION = typer.Option(
    "config.yaml",
    help="Path to ChatGPT CLI config file.",
    envvar="CHATGPTCLI_CONFIG",
)
SYSTEM_OPTION = typer.Option(
    "You are a helpful assistant. Answer as concisely as possible.",
    help="System message for ChatGPT.",
    rich_help_panel="Model parameters",
)
N_OPTION = typer.Option(
    1,
    help="Number of answers for each prompt.",
    min=1,
    max=10,
    rich_help_panel="Model parameters",
)
TEMPERATURE_OPTION = typer.Option(
    1,
    help="Nucleus sampling: higher means more random output.",
    min=0,
    max=2,
    rich_help_panel="Model parameters",
)
TOP_P_OPTION = typer.Option(
    1,
    help="Nucleus sampling: higher means more random output.",
    min=0,
    max=1,
    rich_help_panel="Model parameters",
)
MODEL_OPTION = typer.Option(
    "gpt-3.5-turbo",
    help="Model name, check [here](https://platform.openai.com/docs/models/model-endpoint-compatibility) for alternative models.",
    rich_help_panel="Model parameters",
)
MAX_TOKENS_OPTION = typer.Option(
    2048,
    help="Max number of tokens in completion.",
    rich_help_panel="Model parameters",
)
PRESENCE_PENALTY_OPTION = typer.Option(
    0,
    min=-2,
    max=2,
    help="Penalty for repeating already existing words.",
    rich_help_panel="Model parameters",
)
FREQUENCY_PENALTY_OPTION = typer.Option(
    0,
    min=-2,
    max=2,
    help="Penalty for repeating already frequent words.",
    rich_help_panel="Model parameters",
)
STOP_OPTION = typer.Option(
    None,
    help="Sequences where the API will stop generating further tokens.",
    rich_help_panel="Model parameters",
)


def validate_cli_parameters(
    temperature: float, top_p: float, stop: List[str] | None
) -> Tuple[float, float, List[str] | None]:
    if temperature != 1 and top_p != 1:
        msg = 'OpenAI recommends to change either "temperature" or "top_p", not both.'
        pretty.warning(msg)

    if stop is not None and len(stop) > 4:
        msg = "More than 4 stop sequences provided. Will use the first 4."
        pretty.warning(msg)
        stop = stop[:4]

    return temperature, top_p, stop


@app.command()
def init():
    api_key = rich.prompt.Prompt.ask("OpenAI API key")


@app.command()
def chat(
    config: typer.FileBinaryRead = CONFIG_OPTION,
    model: str = MODEL_OPTION,
    system: str = SYSTEM_OPTION,
    stop: Optional[List[str]] = STOP_OPTION,
    max_tokens: int = MAX_TOKENS_OPTION,
    temperature: float = TEMPERATURE_OPTION,
    top_p: float = TOP_P_OPTION,
    presence_penalty: float = PRESENCE_PENALTY_OPTION,
    frequency_penalty: float = FREQUENCY_PENALTY_OPTION,
):
    if "gpt" not in model:
        msg = Markdown(
            " ".join(
                [
                    "The provided model,",
                    f'"{model}", will probably not work with /v1/chat/completions',
                    "endpoint. Check",
                    "[here](https://platform.openai.com/docs/models/model-endpoint-compatibility)",
                    "the model endpoint compatibility list.",
                ]
            )
        )
        print(msg)
        cont = typer.confirm("Are you sure you want to continue?")
        if not cont:
            raise typer.Abort()

    temperature, top_p, stop = validate_cli_parameters(temperature, top_p, stop)

    config = Config(config)

    chat = Chat(
        config=config,
        system=system,
        model=model,
        stop=stop,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    chat.start()


@app.command()
def prompt(
    config: typer.FileBinaryRead = CONFIG_OPTION,
    model: str = MODEL_OPTION,
    system: str = SYSTEM_OPTION,
    stop: Optional[List[str]] = STOP_OPTION,
    max_tokens: int = MAX_TOKENS_OPTION,
    temperature: float = TEMPERATURE_OPTION,
    top_p: float = TOP_P_OPTION,
    n: int = N_OPTION,
    presence_penalty: float = PRESENCE_PENALTY_OPTION,
    frequency_penalty: float = FREQUENCY_PENALTY_OPTION,
):
    "Can read from stdin."
    stdout = False
    if sys.stdin.isatty():
        # No stdin, show prompt
        prompt = rich.prompt.Prompt
        prompt.prompt_suffix = "> "
        user_input = prompt.ask()
        print()
    else:
        user_input = "\n".join(sys.stdin.readlines())
        stdout = True

    temperature, top_p, stop = validate_cli_parameters(temperature, top_p, stop)

    config = Config(config)

    if "gpt" in model:
        # Use /v1/chat/completions API with GPT-3.5 or GPT-4 models.
        single_prompt = Chat(
            config=config,
            system=system,
            model=model,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        completions = single_prompt.prompt(user_input)
    else:
        if max_tokens > 128:
            cont = typer.confirm(
                "InstructGPT often generates repetitive answers"
                + "exhausting max_tokens. Are you sure want to continue?"
            )
            if not cont:
                raise typer.Abort()
        single_prompt = Prompt(
            config=config,
            model=model,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        completions = single_prompt.prompt(user_input)

    for i, completion in enumerate(completions):
        if len(completions) > 1:
            # Add header for each completions
            if stdout:
                header = "\n".join(["=" * 80, f"Completion {i:d}", "=" * 80])
                print(header)
            else:
                print(Markdown(f"# Completion {i:d}"))
        if stdout:
            print(completion)
        else:
            print(Markdown(completion))


if __name__ == "__main__":
    app()
