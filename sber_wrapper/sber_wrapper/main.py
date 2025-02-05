import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse

from .config import MODELS
from .llm_configs.llm_config_sber import get_sber_config_async
from .helper_functions import process_stream_response, generate_and_get_image

logger = logging.getLogger("uvicorn.error")

CONFIG = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CONFIG
    CONFIG = await get_sber_config_async()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        # Get the request body
        body = await request.json()
        logger.info(
            f"Received chat completion request for model: {body.get('model', 'GigaChat')}"
        )

        # Extract messages and model from the request
        messages = body.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        model_name = body.get("model", "GigaChat")
        stream = body.get("stream", False)

        # Create chat completion using OpenAI client
        if stream:
            response = await CONFIG.async_client.chat.completions.create(
                model=model_name, messages=messages, stream=True
            )
            return StreamingResponse(
                process_stream_response(response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Transfer-Encoding": "chunked",
                },
            )

        # For non-streaming requests
        response = await CONFIG.async_client.chat.completions.create(
            model=model_name, messages=messages
        )

        logger.info(f"Successfully processed request for {model_name}")
        return response

    except Exception as e:
        logger.error(f"Error processing chat completion request: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models in OpenAI format."""
    models_data = []
    timestamp = int(datetime.now().timestamp())

    for model_id in MODELS:
        models_data.append(
            {
                "id": model_id,
                "object": "model",
                "created": timestamp,
                "owned_by": "Sber",
                "permission": [],
                "root": model_id,
                "parent": None,
            }
        )

    return {"data": models_data, "object": "list"}


@app.post("/v1/images/generations")
async def create_image(
    prompt: str = Body(...),
    n: Optional[int] = Body(1),
    size: Optional[str] = Body("1024x1024"),
    style: Optional[str] = Body(None),
):
    """Generate images using Sber API in OpenAI-compatible format."""
    try:
        logger.info(f"Received image generation request: {prompt}")
        # Generate image
        result = await generate_and_get_image(
            client=CONFIG.async_client, prompt=prompt, size=size, style=style
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
