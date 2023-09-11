# ChatGPT CLI

## TODO in this branch

- [X] Add empty line
- [X] Remove too many different commands
- [X] Rename options: max tokens should be clear, --out --history inconsistent.
- [X] Infer default max tokens for context and completion depending on the model
- [X] Rename the app. "chatgpt_cli chat" is ecessive, moreover, gpt-4 is not chat GPT.
- [X] Bump the version.
- [X] Catch `APIConnectionError`
- [X] Rename history into context
- [X] Catch `AuthenticationError` error.
- [X] Add some way to properly uninstall
- [X] Add tests.
- [ ] Show what the model is typing instead of just "Typing" spinner

## Install

The easiest is to install it with [pipx](https://pypa.github.io/pipx/):

```bash
pipx install git+https://gitlab.com/nickto/chatgpt-cli.git
```

## Set up development environment

### Install required Python version with [pyenv](https://github.com/pyenv/pyenv)

You can skip this if you system version is already compatible with the
requirements in `pyproject.toml`.

```bash
pyenv install 3.11.2 # or any other compatible version
```

### Create virtual environment

If you are using system Python:
```bash
/usr/bin/env python -m venv .venv
```

If you are using pyenv:
```bash
$(pyenv shell 3.11.2; python -m venv .venv)
```

### Activate virtual environment

```bash
source .venv/bin/activate
```

and when done working on this project:

```bash
deactivate
```

Alternatively, automate it with [direnv](https://direnv.net/):
```bash
echo 'source .venv/bin/activate\nunset PS1' >> .envrc && direnv allow
```

### Install dependencies

```bash
poetry install
```

Make sure you have installed all the dependencies:
```bash
poetry install | grep -q 'No dependencies to install or update' && echo "All good\!" || echo "Some packages are missing :("
```
