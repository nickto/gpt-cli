import os
from pathlib import Path

CONFIG_DIR = os.path.join(Path.home(), ".config", "gpt-cli")
OPENAI_API_KEY_FILENAME = "openai_api_key"

DEFAULT_SYSTEM = "You are a helpful assistant. Answer as concisely as possible."
