# ChatGPT CLI

## TODO
- [X] single prompt
- [X] more params
- [X] output to file
- [X] implement init
- [X] implement installation
- [X] Capture ~~CTRL + C~~ and exit
- [ ] feed a file to use it as history
- [X] no-warning flag
- [ ] improve token calculations by saving it from response

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
