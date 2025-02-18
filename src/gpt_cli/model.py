from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field, computed_field


class ModelName(str, Enum):
    # Source: https://platform.openai.com/docs/models/gpt-4o
    gpt_4o = "gpt-4o"
    gpt_4o_mini = "gpt-4o-mini"
    gpt_o1 = "o1"
    gpt_o1_mini = "o1-mini"
    gpt_o3_mini = "o3-mini"


class OpenAiModel(BaseModel):
    name: ModelName = Field(default=ModelName.gpt_4o_mini)
    _MAX_OUTPUT_TOKENS: Dict[ModelName, int] = {
        ModelName.gpt_4o: 16_384,
        ModelName.gpt_4o_mini: 16_384,
        ModelName.gpt_o1: 100_000,
        ModelName.gpt_o1_mini: 65_536,
        ModelName.gpt_o3_mini: 100_000,
    }

    _MAX_CONTEXT_TOKENS: Dict[ModelName, int] = {
        ModelName.gpt_4o: 128_000,
        ModelName.gpt_4o_mini: 128_000,
        ModelName.gpt_o1: 200_000,
        ModelName.gpt_o1_mini: 128_000,
        ModelName.gpt_o3_mini: 200_000,
    }

    @computed_field
    @property
    def max_output_tokens(self) -> int:
        return self._MAX_OUTPUT_TOKENS[self.name]

    @computed_field
    @property
    def max_context_tokens(self) -> int:
        return self._MAX_CONTEXT_TOKENS[self.name]
