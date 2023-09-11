from pprint import pprint

from gpt_cli.chat import Context, Role
from gpt_cli.message import Message
from gpt_cli.model import OpenAiModel


def test_with_system():
    model = OpenAiModel(name="gpt-3.5-turbo")
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


def test_without_system():
    model = OpenAiModel(name="gpt-3.5-turbo")
    context = Context(model=model)
    assert len(context.get_messages()) == 0
    context.add_message(Message(content="Who is Banksy?", role=Role.user, model=model))
    assert len(context.get_messages()) == 1
    context.add_message(
        Message(content="I don't know", role=Role.assistant, model=model)
    )
    assert len(context.get_messages()) == 2


def test_limit_messages_with_system():
    model = OpenAiModel(name="gpt-3.5-turbo")
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


def test_limit_messages_without_system():
    model = OpenAiModel(name="gpt-3.5-turbo")
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


def test_limit_tokens():
    model = OpenAiModel(name="gpt-3.5-turbo")
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
    assert len(context.get_messages(max_tokens=15)) == 3  # checked manually, seemed ok


def test_load():
    model = OpenAiModel(name="gpt-3.5-turbo")
    context = Context(model)
    context.load(open("tests/assets/context_w_system.yaml", "r"))

    assert isinstance(context, Context)
    assert len(context.messages) > 0


def test_is_system_set():
    model = OpenAiModel(name="gpt-3.5-turbo")
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
