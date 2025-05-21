import pytest

from gpt_cli.tokens import count_tokens


def test_count_tokens(default_model_for_tests):
    model = default_model_for_tests
    text = "Hello, world!"
    assert count_tokens(text, model.name) == 4
