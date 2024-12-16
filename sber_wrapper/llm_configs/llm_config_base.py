from dataclasses import dataclass
from typing import Optional, Union
from openai import OpenAI, AsyncOpenAI
from gigachat import GigaChat

@dataclass
class LLMConfig:
    client: Union[OpenAI, GigaChat]
    async_client: Optional[AsyncOpenAI]  # Made optional since GigaChat doesn't have async client
    model_name: str
    timeout: int = 120
