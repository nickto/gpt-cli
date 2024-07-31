from enum import Enum
from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ModelName(str, Enum):
    # Source: https://platform.openai.com/docs/models/gpt-4
    gpt_3_5_turbo = "gpt-3.5-turbo"
    gpt_4 = "gpt-4"
    gpt_4_turbo = "gpt-4-turbo"
    gpt_4o = "gpt-4o"
    gpt_4o_mini = "gpt-4o-mini"


class OpenAiModel(BaseModel):
    name: ModelName = Field(default=ModelName.gpt_4o_mini)
    # This is max completion, not max context tokens
    # TODO: allow both values and treat them correctly
    _MAX_TOKENS: Dict[ModelName, int] = {
        ModelName.gpt_3_5_turbo: 4_096,
        ModelName.gpt_4: 8_192,
        ModelName.gpt_4_turbo: 4_096,
        ModelName.gpt_4o: 4_096,
        ModelName.gpt_4o_mini: 16_384,
    }

    @computed_field
    @property
    def max_tokens(self) -> int:
        return self._MAX_TOKENS[self.name]
