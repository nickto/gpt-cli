from gpt_cli.tokens import count_tokens


def test_count_tokens():
    text = "Hello, world!"
    model = "gpt-3.5-turbo"
    assert count_tokens(text, model) == 4
