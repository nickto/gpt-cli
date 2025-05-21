import openai
import pytest
from openai import ChatCompletion

from gpt_cli.key import OpenaiApiKey
from gpt_cli.model import ModelName, OpenAiModel


@pytest.mark.parametrize("model_name", [m for m in ModelName])
def test_all_models_have_max_tokens(model_name):
    model = OpenAiModel(name=model_name)
    assert isinstance(model.max_output_tokens, int)
    assert model.max_output_tokens > 0

    assert isinstance(model.max_context_tokens, int)
    assert model.max_context_tokens > 0


@pytest.mark.parametrize("model_name", [m for m in ModelName])
def test_model_dump(model_name):
    model = OpenAiModel(name=model_name)
    loaded_model = OpenAiModel.model_validate(model.model_dump())
    assert loaded_model == model


@pytest.mark.parametrize("model_name", [m for m in ModelName])
def test_no_exception_when_called(model_name, use_o3):
    openai.api_key = OpenaiApiKey().get()
    if model_name == ModelName.gpt_o3 and not use_o3:
        pytest.skip("Skipping test for o3 model. Use --o3 to run it.")

    response = ChatCompletion.create(
        model=model_name.value,
        messages=[{"role": "user", "content": "DO NOT DO ANYTHING! Just return 'OK'"}],
    )
    assert response["choices"][0]["message"]["content"] == "OK"  # type: ignore
