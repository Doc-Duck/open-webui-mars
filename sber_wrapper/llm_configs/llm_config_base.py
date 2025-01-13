from dataclasses import dataclass

from openai import OpenAI, AsyncOpenAI


@dataclass
class LLMConfig:
    client: OpenAI
    async_client: AsyncOpenAI
    model_name: str
    timeout: int = 120
