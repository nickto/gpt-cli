import pytest

from gpt_cli.model import ModelName, OpenAiModel


@pytest.fixture
def default_model_for_tests():
    return OpenAiModel(name=ModelName.gpt_4_1_nano)  # cheapest
