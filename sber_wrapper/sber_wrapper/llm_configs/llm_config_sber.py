import uuid
import httpx
import requests
import base64
from openai import AsyncOpenAI, OpenAI, AuthenticationError
from openai.types.chat import ChatCompletion

from ..config import API_KEY
from .llm_config_base import LLMConfig


async def generate_sber_token_async():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload = "scope=GIGACHAT_API_CORP"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {API_KEY}",
    }
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(url, headers=headers, data=payload)
        response_json = response.json()
        return response_json["access_token"]


async def download_image_async(file_id: str, token: str) -> bytes:
    """Download image from Sber API using file ID."""
    url = f"https://gigachat.devices.sberbank.ru/api/v1/files/{file_id}/content"
    headers = {"Accept": "application/jpg", "Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.content


class AsyncSberOpenai(AsyncOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_chat_retries()

    def _setup_chat_retries(self):
        original_create = self.chat.completions.create

        async def create_with_retries(*args, **kwargs) -> ChatCompletion:
            is_retry = False
            while True:
                try:
                    return await original_create(*args, **kwargs)
                except AuthenticationError as e:
                    if not is_retry:
                        self.api_key = await generate_sber_token_async()
                        is_retry = True
                    else:
                        error_msg = self._format_error(e)
                        raise Exception(
                            f"Failed after {self._max_retries} attempts. {error_msg}"
                        )

        self.chat.completions.create = create_with_retries


async def get_sber_config_async():
    api_key = await generate_sber_token_async()
    async_llm_client = AsyncSberOpenai(
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
        api_key=api_key,
        http_client=httpx.AsyncClient(verify=False),
    )
    return LLMConfig(
        client=None,
        async_client=async_llm_client,
    )
