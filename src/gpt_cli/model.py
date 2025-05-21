from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field, computed_field


class ModelName(str, Enum):
    # Source: https://platform.openai.com/docs/models
    # (Following the structure on this page)

    # Reasoning models:
    gpt_o4_mini = "o4-mini"
    gpt_o3_mini = "o3-mini"
    gpt_o3 = "o3"
    gpt_o1 = "o1"
    gpt_o1_pro = "o1-pro"

    # Flagship models:
    gpt_4_1 = "gpt-4.1"
    gpt_4o = "gpt-4o"
    # gpt_4o_audio = "gpt-4o-audio-preview" # no audio in our CLI
    gpt_chatgpt_4o = "chatgpt-4o"

    # Cost optimized models:
    # gpt_o4_mini = "o4-mini" # also defined above
    gpt_4_1_nano = "gpt-4.1-nano"
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4_1_mini = "gpt-4.1-mini"
    # gpt_o3_mini = "o3-mini" # also defined above
    # gpt_4o_mini_audio = "gpt-4o-mini-audio-preview" # no audio in our CLI


class OpenAiModel(BaseModel):
    name: ModelName = Field(default=ModelName.gpt_4o_mini)
    _MAX_CONTEXT_TOKENS: Dict[ModelName, int] = {
        ModelName.gpt_4o_mini: 200_000,
        ModelName.gpt_o3: 200_000,
        ModelName.gpt_o3_mini: 200_000,
        ModelName.gpt_o1: 200_000,
        ModelName.gpt_o1_pro: 200_000,
        ModelName.gpt_4_1: 1_047_576,
        ModelName.gpt_4o: 128_000,
        # ModelName.gpt_4o_audio: 128_000,
        ModelName.gpt_chatgpt_4o: 128_000,
        ModelName.gpt_o4_mini: 200_000,
        ModelName.gpt_4_1_mini: 1_047_576,
        ModelName.gpt_4_1_nano: 1_047_576,
        ModelName.gpt_4o_mini: 128_000,
        # ModelName.gpt_4o_mini_audio: 128_000,
    }

    _MAX_OUTPUT_TOKENS: Dict[ModelName, int] = {
        ModelName.gpt_4o_mini: 100_000,
        ModelName.gpt_o3: 100_000,
        ModelName.gpt_o3_mini: 100_000,
        ModelName.gpt_o1: 100_000,
        ModelName.gpt_o1_pro: 100_000,
        ModelName.gpt_4_1: 32_768,
        ModelName.gpt_4o: 16_384,
        # ModelName.gpt_4o_audio: 16_384,
        ModelName.gpt_chatgpt_4o: 16_384,
        ModelName.gpt_o4_mini: 100_000,
        ModelName.gpt_4_1_mini: 32_768,
        ModelName.gpt_4_1_nano: 32_768,
        ModelName.gpt_4o_mini: 16_384,
        # ModelName.gpt_4o_mini_audio: 16_384,
    }

    @computed_field
    @property
    def max_output_tokens(self) -> int:
        return self._MAX_OUTPUT_TOKENS[self.name]

    @computed_field
    @property
    def max_context_tokens(self) -> int:
        return self._MAX_CONTEXT_TOKENS[self.name]
