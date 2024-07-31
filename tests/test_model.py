from gpt_cli.model import ModelName, OpenAiModel


def test_all_models_have_max_tokens():
    for model_name in ModelName:
        model = OpenAiModel(name=model_name)
        assert isinstance(model.max_output_tokens, int)
        assert model.max_output_tokens > 0

        assert isinstance(model.max_context_tokens, int)
        assert model.max_context_tokens > 0


def test_model_dump():
    for model_name in ModelName:
        model = OpenAiModel(name=model_name)
        loaded_model = OpenAiModel.model_validate(model.model_dump())
        assert loaded_model == model
