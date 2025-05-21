import tiktoken
from gpt_cli.model import ModelName
import re


def guess_encoding_using_heuristics(model_name: ModelName) -> tiktoken.Encoding:
    # oX models, e.g. o3, o4, o1, o1-pro
    if re.search(r"o\d+", model_name.value):
        return tiktoken.get_encoding("o200k_base")
    # gpt-4.1, gpt-4o, etc. But NOT plain gpt-4, or gpt-4-turbo
    if re.search(r"gpt-4(?:\.[1-9]\d*|o.*)", model_name.value):
        return tiktoken.get_encoding("o200k_base")

    raise ValueError(f"Could not guess encoding for model {model_name.value}.")


def count_tokens(text: str, model: ModelName) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model.value)
    except (KeyError, ValueError):
        encoding = guess_encoding_using_heuristics(model_name=model)

    num_tokens = len(encoding.encode(text))
    return num_tokens
