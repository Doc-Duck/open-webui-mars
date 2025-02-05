from dataclasses import dataclass

from openai import OpenAI, AsyncOpenAI


@dataclass
class LLMConfig:
    client: OpenAI
    async_client: AsyncOpenAI
    timeout: int = 120
