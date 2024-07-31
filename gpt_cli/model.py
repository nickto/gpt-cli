from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field, computed_field


class ModelName(str, Enum):
    # Source: https://platform.openai.com/docs/models/gpt-4
    gpt_3_5_turbo = "gpt-3.5-turbo"
    gpt_4 = "gpt-4"
    gpt_4_turbo = "gpt-4-turbo"
    gpt_4o = "gpt-4o"
    gpt_4o_mini = "gpt-4o-mini"


class OpenAiModel(BaseModel):
    name: ModelName = Field(default=ModelName.gpt_4o_mini)
    _MAX_OUTPUT_TOKENS: Dict[ModelName, int] = {
        ModelName.gpt_3_5_turbo: 4_096,
        ModelName.gpt_4: 8_192,
        ModelName.gpt_4_turbo: 4_096,
        ModelName.gpt_4o: 4_096,
        ModelName.gpt_4o_mini: 16_384,
    }

    _MAX_CONTEXT_TOKENS: Dict[ModelName, int] = {
        ModelName.gpt_3_5_turbo: 16_385,
        ModelName.gpt_4: 8_192,
        ModelName.gpt_4_turbo: 128_000,
        ModelName.gpt_4o: 128_000,
        ModelName.gpt_4o_mini: 128_000,
    }

    @computed_field
    @property
    def max_output_tokens(self) -> int:
        return self._MAX_OUTPUT_TOKENS[self.name]

    @computed_field
    @property
    def max_context_tokens(self) -> int:
        return self._MAX_CONTEXT_TOKENS[self.name]
