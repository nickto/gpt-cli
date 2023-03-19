import os
import sys
from typing import Dict, List, Optional, Tuple

import rich
import typer
from rich.markdown import Markdown

from chatgpt_cli import pretty
from chatgpt_cli.chat import Chat, History
from chatgpt_cli.constants import CONFIG_DIR
from chatgpt_cli.instruct import Instruct
from chatgpt_cli.key import OpenaiApiKey

app = typer.Typer(rich_markup_mode="markdown")


API_KEY_OPTION = typer.Option(
    None,
    help="OpenAI API key (run `chatgpt-cli init` to avoid passing it each time).",
    show_default=False,
    envvar="OPENAI_API_KEY",
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
OUTPUT_OPTION = typer.Option(
    None,
    "--out",
    "-o",
    help="Output the whole conversation to a file.",
    metavar="PATH",
    show_default=False,
)
NOWARNING_OPTION = typer.Option(
    False,
    "--nowarning",
    help="Do not show warnings.",
)
NOCONFIRM_OPTION = typer.Option(
    False,
    "--noconfirm",
    help="Answer yes to all confirmation messages.",
)
PROMPT_ARGUMENT = typer.Argument(
    "",
    help="Prompt for a model.",
    metavar="TEXT",
)
HISTORY_OPTION = typer.Option(
    None,
    help="Path to previous outputs, to use as chat history.",
    show_default=False,
)


def validate_cli_parameters(
    temperature: float,
    top_p: float,
    stop: List[str] | None,
    nowarning: bool = False,
) -> Tuple[float, float, List[str] | None]:
    if temperature != 1 and top_p != 1 and not nowarning:
        msg = 'OpenAI recommends to change either "temperature" or "top_p", not both.'
        pretty.warning(msg)

    if stop is not None and len(stop) > 4 and not nowarning:
        msg = "More than 4 stop sequences provided. Will use the first 4."
        pretty.warning(msg)
        stop = stop[:4]

    return temperature, top_p, stop


@app.command()
def init(noconfirm: bool = NOCONFIRM_OPTION):
    "Initialize the app: provide it with an OpenAI API key."
    # Get API key
    openai_api_key = rich.prompt.Prompt.ask("OpenAI API key")

    # Get config path and check if it exists
    api_key_path = OpenaiApiKey.path
    cont = typer.confirm(f"Will save OpenAI API key to {api_key_path}, okay?")
    if not cont:
        raise typer.Abort()

    if os.path.exists(api_key_path) and not noconfirm:
        cont = typer.confirm(
            f"{api_key_path} file already exists. Do you want to overwrite it?"
        )
        if not cont:
            raise typer.Abort()

    # Save
    os.makedirs(os.path.split(api_key_path)[0], exist_ok=True)
    OpenaiApiKey(openai_api_key=openai_api_key).save()
    rich.print(f"OpenAI API key was saved to {api_key_path} successfully.")


@app.command()
def chat(
    openai_api_key: str = API_KEY_OPTION,
    out: str = OUTPUT_OPTION,
    model: str = MODEL_OPTION,
    system: str = SYSTEM_OPTION,
    stop: Optional[List[str]] = STOP_OPTION,
    max_tokens: int = MAX_TOKENS_OPTION,
    temperature: float = TEMPERATURE_OPTION,
    top_p: float = TOP_P_OPTION,
    presence_penalty: float = PRESENCE_PENALTY_OPTION,
    frequency_penalty: float = FREQUENCY_PENALTY_OPTION,
    noconfirm: bool = NOCONFIRM_OPTION,
    nowarning: bool = NOWARNING_OPTION,
    history: typer.FileText = HISTORY_OPTION,
):
    "Start an interactive chat."
    openai_api_key = OpenaiApiKey(openai_api_key)

    if "gpt" not in model and not noconfirm:
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
        rich.print(msg)
        cont = typer.confirm("Are you sure you want to continue?")
        if not cont:
            raise typer.Abort()

    if history:
        history = History(model=model).load(history)
        # Check that there are no contradictions between the system in the
        # history and the history supplied via command line
        if history.system.content is not None and system is not None:
            msg = (
                "ignoring system from --system parameter: system"
                "message present in history."
            )
            if not nowarning:
                pretty.warning(msg)

    temperature, top_p, stop = validate_cli_parameters(
        temperature, top_p, stop, nowarning
    )
    chat = Chat(
        api_key=openai_api_key,
        out=out,
        system=system,
        model=model,
        stop=stop,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        history=history,
    )
    chat.start()


@app.command()
def prompt(
    openai_api_key: str = API_KEY_OPTION,
    out: str = OUTPUT_OPTION,
    model: str = MODEL_OPTION,
    system: str = SYSTEM_OPTION,
    stop: Optional[List[str]] = STOP_OPTION,
    max_tokens: int = MAX_TOKENS_OPTION,
    temperature: float = TEMPERATURE_OPTION,
    top_p: float = TOP_P_OPTION,
    n: int = N_OPTION,
    presence_penalty: float = PRESENCE_PENALTY_OPTION,
    frequency_penalty: float = FREQUENCY_PENALTY_OPTION,
    noconfirm: bool = NOCONFIRM_OPTION,
    nowarning: bool = NOWARNING_OPTION,
    prompt: Optional[str] = PROMPT_ARGUMENT,
):
    """
    Ask a single question.

    Checks for prompt in the command line argument, then in standard input.
    If neither is present, asks interactively.
    """
    openai_api_key = OpenaiApiKey(openai_api_key)

    if prompt:
        user_input = prompt
        del prompt
    else:
        if not sys.stdin.isatty():
            # Read from piped stdin
            user_input = "\n".join(sys.stdin.readlines())
        else:
            prompt = rich.prompt.Prompt
            prompt.prompt_suffix = "> "
            user_input = prompt.ask()
            rich.print()

    if not sys.stdout.isatty():
        # Write to piped stdout
        print("User:")
        print(user_input)

    if out:
        with open(out, "w+") as f:
            f.write("User:\n")
            f.write(user_input + "\n\n")

    temperature, top_p, stop = validate_cli_parameters(
        temperature, top_p, stop, nowarning
    )
    kwargs = dict(
        api_key=openai_api_key,
        model=model,
        stop=stop,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    if "gpt" in model:
        # Use /v1/chat/completions API with GPT-3.5 or GPT-4 models.
        kwargs["system"] = system
        single_prompt = Chat(**kwargs)
        completions = single_prompt.prompt(user_input)
    else:
        if max_tokens > 128 and not noconfirm:
            cont = typer.confirm(
                "InstructGPT often generates repetitive answers "
                "exhausting max_tokens. Are you sure want to continue?"
            )
            if not cont:
                raise typer.Abort()
        single_prompt = Instruct(**kwargs)
        completions = single_prompt.prompt(user_input)

    for i, completion in enumerate(completions):
        if len(completions) == 1:
            header = "Assistant:"
            header_md = ""
        else:
            header = f"Assistant, completion {i:d}:"
            header_md = f"# Completion {i:d}\n"

        if sys.stdout.isatty():
            # Terminal output, so format
            rich.print(Markdown(header_md), end="")
            rich.print(Markdown(completion))
        else:
            # Stdout output, skip formatting
            print(header)
            print(completion)

        if out:
            with open(out, "a+") as f:
                f.write(header + "\n")
                f.write(completion + "\n\n")
