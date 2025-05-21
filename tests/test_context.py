import tempfile

import pytest

from gpt_cli.chat import Context, Role
from gpt_cli.message import Message
from gpt_cli.model import OpenAiModel, ModelName


@pytest.fixture
def save_filepath():
    # Get temporary file path
    with tempfile.NamedTemporaryFile() as f:
        filepath = f.name
        print(filepath)
        yield filepath


def test_with_system(default_model_for_tests):
    model = default_model_for_tests
    context = Context(model=model)
    assert len(context.get_messages()) == 0
    context.set_system("You are a helpful assistant.")
    assert len(context.get_messages()) == 1
    context.add_message(Message(content="Who is Banksy?", role=Role.user, model=model))
    assert len(context.get_messages()) == 2
    context.add_message(
        Message(content="I don't know", role=Role.assistant, model=model)
    )
    assert len(context.get_messages()) == 3


def test_without_system(default_model_for_tests):
    model = default_model_for_tests
    context = Context(model=model)
    assert len(context.get_messages()) == 0
    context.add_message(Message(content="Who is Banksy?", role=Role.user, model=model))
    assert len(context.get_messages()) == 1
    context.add_message(
        Message(content="I don't know", role=Role.assistant, model=model)
    )
    assert len(context.get_messages()) == 2


def test_limit_messages_with_system(default_model_for_tests):
    model = default_model_for_tests
    context = Context(model=model)
    context.set_system("You are a helpful assistant.")
    assert len(context.get_messages()) == 1
    context.add_message(Message(content="Who is Banksy?", role=Role.user, model=model))
    assert len(context.get_messages()) == 2
    context.add_message(
        Message(content="I don't know", role=Role.assistant, model=model)
    )
    assert len(context.get_messages()) == 3
    context.add_message(Message(content="Are you sure?", role=Role.user, model=model))
    assert len(context.get_messages()) == 4
    context.add_message(
        Message(content="Yes I am sure.", role=Role.assistant, model=model)
    )
    assert len(context.get_messages()) == 5
    assert len(context.get_messages(max_messages=2)) == 3  # 3 because have system


def test_limit_messages_without_system(default_model_for_tests):
    model = default_model_for_tests
    context = Context(model=model)
    assert len(context.get_messages()) == 0
    context.add_message(Message(content="Who is Banksy?", role=Role.user, model=model))
    assert len(context.get_messages()) == 1
    context.add_message(
        Message(content="I don't know", role=Role.assistant, model=model)
    )
    assert len(context.get_messages()) == 2
    context.add_message(Message(content="Are you sure?", role=Role.user, model=model))
    assert len(context.get_messages()) == 3
    context.add_message(
        Message(content="Yes I am sure.", role=Role.assistant, model=model)
    )
    assert len(context.get_messages(max_messages=2)) == 2


def test_limit_tokens(default_model_for_tests):
    model = default_model_for_tests
    context = Context(model=model)
    context.set_system("You are a helpful assistant.")
    assert len(context.get_messages()) == 1
    context.add_message(Message(content="Who is Banksy?", role=Role.user, model=model))
    assert len(context.get_messages()) == 2
    context.add_message(
        Message(content="I don't know", role=Role.assistant, model=model)
    )
    assert len(context.get_messages()) == 3
    context.add_message(Message(content="Are you sure?", role=Role.user, model=model))
    assert len(context.get_messages()) == 4
    context.add_message(
        Message(content="Yes I am sure.", role=Role.assistant, model=model)
    )
    assert (
        len(context.get_messages(max_context_tokens=15)) == 3
    )  # checked manually, seemed ok


def test_load(default_model_for_tests):
    model = default_model_for_tests
    context = Context(model)
    context.load(open("tests/assets/context_w_system.yaml", "r"))

    assert isinstance(context, Context)
    assert len(context.messages) > 0


def test_is_system_set(default_model_for_tests):
    model = default_model_for_tests
    context = Context(model)

    context.load(open("tests/assets/context_w_system.yaml", "r"))

    assert context.is_system_set()
    assert context.system.content is not None

    assert context.system.role == Role.system
    assert context.messages[0].role != Role.system

    context = Context(model)
    context.load(open("tests/assets/context_wo_system.yaml", "r"))

    assert not context.is_system_set()
    assert context.system.content is None


def test_save(save_filepath, default_model_for_tests):
    model = default_model_for_tests
    context = Context(model=model)
    assert len(context.get_messages()) == 0
    context.set_system("You are a helpful assistant.")
    assert len(context.get_messages()) == 1
    context.add_message(Message(content="Who is Banksy?", role=Role.user, model=model))
    assert len(context.get_messages()) == 2
    context.add_message(
        Message(content="I don't know", role=Role.assistant, model=model)
    )
    context.save(save_filepath)

    # Make sure loaded content is the same
    loaded_context = Context(model=model).load(save_filepath)
    assert context.system.content == loaded_context.system.content
    for m1, m2 in zip(context.messages, loaded_context.messages):
        assert m1.content == m2.content
        assert m1.role == m2.role
