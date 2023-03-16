# ChatGPT CLI

## Set up development environment

```bash
make .venv && echo 'source .venv/bin/activate\nunset PS1' >> .envrc && direnv allow
```

## Example usage

Start
```bash
chatgpt-cli                                             # continue previous session
chatgpt-cli --new                                       # start new session
chatgpt-cli --new --service "You are helpful assistant" # start new sessions ans specify service message
```