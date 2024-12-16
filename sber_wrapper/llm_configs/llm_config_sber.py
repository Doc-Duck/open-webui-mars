from gigachat import GigaChat
from llm_configs.llm_config_base import LLMConfig

def get_sber_config(model_name: str):
    try:
        # Initialize GigaChat client with credentials
        gigachat_client = GigaChat(
            credentials="NjgyMDc2NWEtYjZiNy00ZTdkLTkwNDYtYTdkNDFkYTU0ZGVhOjEzOTI3M2I2LTU2ZGItNDM2My05ZDhhLTcwNTk3NWVhZjRiOQ==",
            scope="GIGACHAT_API_PERS",
            model=model_name,
            verify_ssl_certs=False
        )
        
        return LLMConfig(
            client=gigachat_client,
            async_client=None,  # GigaChat doesn't have async client yet
            model_name=model_name,
        )
    except Exception as e:
        raise Exception(f"Failed to initialize Sber config: {str(e)}")
