from enum import Enum
from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ModelName(str, Enum):
    # Source: https://platform.openai.com/docs/models/gpt-4
    gpt_4 = "gpt-4"
    gpt_4_32k = "gpt-4-32k"
    gpt_3_5_turbo = "gpt-3.5-turbo"
    gpt_3_5_turbo_16k = "gpt-3.5-turbo-16k"


class OpenAiModel(BaseModel):
    name: ModelName = Field(default=ModelName.gpt_3_5_turbo)
    _MAX_TOKENS: Dict[ModelName, int] = {
        ModelName.gpt_4: 8_192,
        ModelName.gpt_4_32k: 32_768,
        ModelName.gpt_3_5_turbo: 4_096,
        ModelName.gpt_3_5_turbo_16k: 16_384,
    }

    @computed_field
    @property
    def max_tokens(self) -> int:
        return self._MAX_TOKENS[self.name]
