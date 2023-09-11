from gpt_cli.message import Message, Role, OpenAiModel
import pytest


def assert_instance_types(message: Message):
    assert isinstance(message, Message)
    assert isinstance(message.role, Role)
    assert isinstance(message.content, str) or message.content is None
    assert isinstance(message.model, OpenAiModel)
    assert isinstance(message.n_tokens, int)
    assert message.n_tokens >= 0


def test_valid():
    message = Message(role=Role.system, content="You are bot.", model=OpenAiModel())
    assert_instance_types(message)

    message = Message(role=Role.user, content="Hello!", model=OpenAiModel())
    assert_instance_types(message)

    message = Message(role=Role.assistant, content="Hello!", model=OpenAiModel())
    assert_instance_types(message)


def test_none_content():
    message = Message(role=Role.system, content=None, model=OpenAiModel())
    assert_instance_types(message)

    with pytest.raises(ValueError):
        message = Message(role=Role.user, content=None, model=OpenAiModel())

    with pytest.raises(ValueError):
        message = Message(role=Role.assistant, content=None, model=OpenAiModel())
