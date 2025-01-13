import uuid
import httpx
import requests
from openai import AsyncOpenAI, OpenAI

from config import API_KEY
from llm_configs.llm_config_base import LLMConfig


def generate_sber_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload = "scope=GIGACHAT_API_CORP"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {API_KEY}",
    }
    response = requests.request(
        "POST", url, headers=headers, data=payload, verify=False
    )
    return response.json()["access_token"]


def get_sber_config(model_name: str):
    api_key = generate_sber_token()
    print(api_key)
    sber_llm_client = OpenAI(
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
        api_key=api_key,
        http_client=httpx.Client(verify=False),
    )

    async_llm_client = AsyncOpenAI(
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
        api_key=api_key,
        http_client=httpx.AsyncClient(verify=False),
    )
    return LLMConfig(
        client=sber_llm_client,
        async_client=async_llm_client,
        model_name=model_name,
    )
