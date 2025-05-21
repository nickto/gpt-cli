import pytest

from gpt_cli.model import ModelName, OpenAiModel


@pytest.fixture
def default_model_for_tests():
    return OpenAiModel(name=ModelName.gpt_4_1_nano)  # cheapest


def pytest_addoption(parser):
    parser.addoption(
        "--o3",
        action="store_true",
        default=False,
        help="run tests on o3",
    )


@pytest.fixture
def use_o3(request):
    return request.config.getoption("--o3")
