import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llm_configs.llm_config_sber import get_sber_config
import json
import asyncio
import logging
import sys
from typing import AsyncGenerator
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configurations
MODELS = {
    "GigaChat": "GigaChat",
    "GigaChat-Pro": "GigaChat-Pro",
    "GigaChat-Max": "GigaChat-Max",
}


async def process_stream_response(stream) -> AsyncGenerator[str, None]:
    """Process streaming response from OpenAI API."""
    try:
        async for chunk in stream:
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
            await asyncio.sleep(0)  # Allow other tasks to run

        # Send the final [DONE] message
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in stream processing: {str(e)}")
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


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

        model_id = body.get("model", "GigaChat")
        config = get_sber_config(MODELS.get(model_id))

        # Check if streaming is requested
        stream = body.get("stream", False)

        # Create chat completion using OpenAI client
        if stream:
            response = await config.async_client.chat.completions.create(
                model=config.model_name, messages=messages, stream=True
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
        response = await config.async_client.chat.completions.create(
            model=config.model_name, messages=messages
        )

        logger.info(f"Successfully processed request for {model_id}")
        return response

    except Exception as e:
        logger.error(f"Error processing chat completion request: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models in OpenAI format."""
    try:
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
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
