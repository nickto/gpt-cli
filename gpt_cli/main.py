import os
from typing import Dict, List, Optional, Tuple, Annotated
from pydantic import ValidationError

import rich
import typer
from rich.markdown import Markdown

from gpt_cli import pretty
from .chat import Chat, Context
from .key import OpenaiApiKey
from .model import OpenAiModel

app = typer.Typer(rich_markup_mode="markdown")


INPUT_OPTION = typer.Option(
    None,
    help="Previous outputs to start the conversation from.",
    show_default=False,
    rich_help_panel="Conversation context",
)
OUTPUT_OPTION = typer.Option(
    None,
    help="Output the whole conversation to a file.",
    metavar="PATH",
    show_default=False,
    rich_help_panel="Conversation context",
)
MAX_CONTEXT_TOKENS_OPTION = typer.Option(
    2048,
    help="Max number of tokens in the context.",
    rich_help_panel="Conversation context",
    show_default=False,
)
API_KEY_OPTION = typer.Option(
    None,
    help="OpenAI API key (run `gpt-cli init` to avoid passing it each time).",
    show_default=False,
    envvar="OPENAI_API_KEY",
    rich_help_panel="Authentication",
)
SYSTEM_OPTION = typer.Option(
    None,
    help="System message for ChatGPT.",
    rich_help_panel="Model parameters",
    show_default=False,
)
TEMPERATURE_OPTION = typer.Option(
    1,
    help="Temperature sampling: higher means more random output.",
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
    show_default=False,
)
MAX_COMPLETION_TOKENS_OPTION = typer.Option(
    2048,
    help="Max number of tokens in the completion.",
    rich_help_panel="Model parameters",
    show_default=False,
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


def validate_model_parameters(
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

    if not noconfirm:
        cont = typer.confirm(f"OpenAI API key will be saved to {api_key_path}, okay?")
        if not cont:
            raise typer.Abort()

    # Check if we need to overwrite it
    if os.path.exists(api_key_path) and not noconfirm:
        cont = typer.confirm(
            f"File {api_key_path} already exists. Do you want to overwrite it?"
        )
        if not cont:
            raise typer.Abort()

    # Save
    os.makedirs(os.path.split(api_key_path)[0], exist_ok=True)
    OpenaiApiKey(openai_api_key=openai_api_key).save()


@app.command()
def deinit(noconfirm: bool = NOCONFIRM_OPTION):
    "Deinitialize the app: remove the OpenAI API key."
    # Get config path and check if it exists
    api_key_path = OpenaiApiKey.path

    # Check if it exists
    if not os.path.exists(api_key_path):
        pretty.warning(f"File {api_key_path} does not exist: nothing to remove.")

    # Ask for confirmations
    if not noconfirm:
        cont = typer.confirm(
            f"OpenAI API key will be removed from {api_key_path}, okay?"
        )
        if not cont:
            raise typer.Abort()

    # Remove
    os.remove(api_key_path)


@app.command()
def chat(
    input: typer.FileText = INPUT_OPTION,
    output: str = OUTPUT_OPTION,
    max_context_tokens: Optional[int] = MAX_CONTEXT_TOKENS_OPTION,
    system: Optional[str] = SYSTEM_OPTION,
    model: str = MODEL_OPTION,
    stop: Optional[List[str]] = STOP_OPTION,
    max_completion_tokens: Optional[int] = MAX_COMPLETION_TOKENS_OPTION,
    temperature: float = TEMPERATURE_OPTION,
    top_p: float = TOP_P_OPTION,
    presence_penalty: float = PRESENCE_PENALTY_OPTION,
    frequency_penalty: float = FREQUENCY_PENALTY_OPTION,
    nowarning: bool = NOWARNING_OPTION,
    openai_api_key: str = API_KEY_OPTION,
):
    """Start an interactive chat.

    For multiline inputs use backslashes. Type "exit" or press Ctrl + C to exit the chat.
    """
    openai_api_key = OpenaiApiKey(openai_api_key)

    try:
        model = OpenAiModel(name=model)
    except ValidationError:
        pretty.error(
            f'Model "{model}" is not supported. '
            "Check the list of supported models here: "
            "https://platform.openai.com/docs/models/model-endpoint-compatibility."
        )
        raise typer.Abort()

    # Load context if provided
    if input:
        try:
            context = Context(model=model).load(input)
        except ValueError as e:
            pretty.error(e)
            raise typer.Abort()

        # Check that there are no contradictions between the system in the
        # history and the history supplied via command line
        if context.is_system_set() and system is not None:
            msg = "Ignoring system from history because system was provided via command line."
            if not nowarning:
                pretty.warning(msg)

            context.set_system(system)
            rich.print(context.system)
    else:
        context = None

    # Validate model parameters, so that they do not contradict each other
    temperature, top_p, stop = validate_model_parameters(
        temperature, top_p, stop, nowarning
    )

    # Create the chat
    chat = Chat(
        api_key=openai_api_key,
        out=output,
        system=system,
        model=model,
        stop=stop,
        max_completion_tokens=max_completion_tokens,
        max_context_tokens=max_context_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        context=context,
    )
    chat.start()


if __name__ == "__main__":
    app()
