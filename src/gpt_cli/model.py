from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field, computed_field


class ModelName(str, Enum):
    # Source: https://platform.openai.com/docs/models

    # 5:
    gpt_5 = "gpt-5"
    gpt_5_mini = "gpt-5-mini"
    gpt_5_nano = "gpt-5-nano"
    gpt_5_chat = "gpt-5-chat-latest"

    # 4.1
    gpt_4_1 = "gpt-4.1"
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_4_1_nano = "gpt-4.1-nano"

    # o3
    gpt_o3 = "o3"
    gpt_o3_deep_research = "o3-deep-research"
    gpt_o3_pro = "o3-pro"
    gpt_o3_mini = "o3-mini"

    # o4
    gpt_o4_mini = "o4-mini"
    # gpt_o4_mini_deep_research = "o4-mini-deep-research"  # only supports v1/responses

    # o1
    gpt_o1 = "o1"
    # gpt_o1_pro = "o1-pro"  # only supports v1/responses

    # 4o
    gpt_4o = "gpt-4o"
    gpt_4o_search_preview = "gpt-4o-search-preview"
    gpt_4o_mini = "gpt-4o-mini"
    gpt_chatgpt_4o = "chatgpt-4o-latest"


class OpenAiModel(BaseModel):
    name: ModelName = Field(default=ModelName.gpt_4o_mini)
    _MAX_CONTEXT_TOKENS: Dict[ModelName, int] = {
        # 5
        ModelName.gpt_5: 400_000,
        ModelName.gpt_5_mini: 400_000,
        ModelName.gpt_5_nano: 400_000,
        ModelName.gpt_5_chat: 128_000,
        # 4.1
        ModelName.gpt_4_1: 1_047_576,
        ModelName.gpt_4_1_mini: 1_047_576,
        ModelName.gpt_4_1_nano: 1_047_576,
        # o3
        ModelName.gpt_o3: 200_000,
        ModelName.gpt_o3_deep_research: 200_000,
        ModelName.gpt_o3_mini: 200_000,
        ModelName.gpt_o3_pro: 200_000,
        # o4
        ModelName.gpt_o4_mini: 200_000,
        # ModelName.gpt_o4_mini_deep_research: 200_000,
        # o1
        ModelName.gpt_o1: 200_000,
        # ModelName.gpt_o1_pro: 200_000,
        # 4o
        ModelName.gpt_4o: 128_000,
        ModelName.gpt_4o_search_preview: 128_000,
        ModelName.gpt_4o_mini: 200_000,
        ModelName.gpt_chatgpt_4o: 128_000,
        ModelName.gpt_4o_mini: 128_000,
    }

    _MAX_OUTPUT_TOKENS: Dict[ModelName, int] = {
        # 5
        ModelName.gpt_5: 128_000,
        ModelName.gpt_5_mini: 128_000,
        ModelName.gpt_5_nano: 128_000,
        ModelName.gpt_5_chat: 16_384,
        # 4.1
        ModelName.gpt_4_1: 32_768,
        ModelName.gpt_4_1_mini: 32_768,
        ModelName.gpt_4_1_nano: 32_768,
        # o3
        ModelName.gpt_o3: 100_000,
        ModelName.gpt_o3_deep_research: 100_000,
        ModelName.gpt_o3_mini: 100_000,
        ModelName.gpt_o3_pro: 100_000,
        # o4
        ModelName.gpt_o4_mini: 100_000,
        # ModelName.gpt_o4_mini_deep_research: 100_000,
        # o1
        ModelName.gpt_o1: 100_000,
        # ModelName.gpt_o1_pro: 100_000,
        # 4o
        ModelName.gpt_4o: 16_384,
        ModelName.gpt_4o_search_preview: 16_384,
        ModelName.gpt_4o_mini: 100_000,
        ModelName.gpt_chatgpt_4o: 16_384,
        ModelName.gpt_4o_mini: 16_384,
    }

    @computed_field
    @property
    def max_output_tokens(self) -> int:
        return self._MAX_OUTPUT_TOKENS[self.name]

    @computed_field
    @property
    def max_context_tokens(self) -> int:
        return self._MAX_CONTEXT_TOKENS[self.name]
