from gpt_cli.model import OpenAiModel, ModelName


def test_all_models_have_max_tokens():
    for model_name in ModelName:
        model = OpenAiModel(name=model_name)
        assert isinstance(model.max_tokens, int)
        assert model.max_tokens > 0
