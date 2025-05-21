# GPT CLI

## Why?

My terminal is always at my fingertips, unlike my browser. Moreover, it allows

- tinkering with some model parameters,
- saving and loading conversation context,
- using latest OpenAI models without paying 20$ a month.

And, to be honest, I also just wanted to play around with OpenAI API and
[Typer](https://typer.tiangolo.com/).

## TODO

- [X] Show what the model is typing instead of just "Typing" spinner
- [ ] Add support for `/v1/responses`, not only `/v1/chat/completions`.
- [ ] Improve README with usage demos.

## Install

The easiest is to install it with [pipx](https://pypa.github.io/pipx/) or [uv tools](https://docs.astral.sh/uv/#tools):

```bash
pipx install git+https://gitlab.com/nickto/gpt-cli.git
```

```bash
uv tool install https://gitlab.com/nickto/gpt-cli.git
```

## Use

TODO

## Set up development environment

### Use [Nix](https://nix.dev/manual/nix/2.18/command-ref/nix-shell) and [direnv](https://direnv.net/)

Create the following `.envrc` file

```
use nix
```

Then run `direnv allow`. This should handle the setup.

### Use [uv](https://docs.astral.sh/uv/) explicitly

```bash
uv venv .venv --python 3.11
```

### Activate virtual environment

```bash
source .venv/bin/activate
```

and when done working on this project:

```bash
deactivate
```

### Install dependencies

```bash
uv sync
```
