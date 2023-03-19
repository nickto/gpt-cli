from chatgpt_cli.chat import History, Role
from pprint import pprint


def test_with_system():
    history = History(model="gpt-3.5-turbo")
    assert len(history.get_history()) == 0
    history.add_system("You are a helpful assistant.")
    assert len(history.get_history()) == 1
    history.add_user("Who is Banksy?")
    assert len(history.get_history()) == 2
    history.add_assistant("I don't know")
    assert len(history.get_history()) == 3


def test_without_system():
    history = History(model="gpt-3.5-turbo")
    history.add_user("Who is Banksy?")
    assert len(history.get_history()) == 1
    history.add_assistant("I don't know")
    assert len(history.get_history()) == 2


def test_limit_messages_with_system():
    history = History(model="gpt-3.5-turbo")
    history.add_system("You are a helpful assistant.")
    assert len(history.get_history()) == 1
    history.add_user("Who is Banksy?")
    assert len(history.get_history()) == 2
    history.add_assistant("I don't know")
    assert len(history.get_history()) == 3
    history.add_user("Are you sure?")
    assert len(history.get_history()) == 4
    history.add_assistant("Yes I am sure.")
    assert len(history.get_history(max_messages=2)) == 3  # 3 because have system


def test_limit_messages_without_system():
    history = History(model="gpt-3.5-turbo")
    assert len(history.get_history()) == 0
    history.add_user("Who is Banksy?")
    assert len(history.get_history()) == 1
    history.add_assistant("I don't know")
    assert len(history.get_history()) == 2
    history.add_user("Are you sure?")
    assert len(history.get_history()) == 3
    history.add_assistant("Yes I am sure.")
    assert len(history.get_history(max_messages=2)) == 2


def test_limit_tokens():
    history = History(model="gpt-3.5-turbo")
    history.add_system("You are a helpful assistant.")
    assert len(history.get_history()) == 1
    history.add_user("Who is Banksy?")
    assert len(history.get_history()) == 2
    history.add_assistant("I don't know")
    assert len(history.get_history()) == 3
    history.add_user("Are you sure?")
    assert len(history.get_history()) == 4
    history.add_assistant("Yes I am sure.")
    assert len(history.get_history(max_tokens=15)) == 3  # checked manually, seemed ok


def test_load():
    history = History(model="gpt-3.5-turbo")
    history.load(open("tests/assets/history.txt", "r"))
    assert history.is_system_set()
    assert history.system.content is not None

    assert history.system.role == Role.system
    assert history.messages[0].role != Role.system
