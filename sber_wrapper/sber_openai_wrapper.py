from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llm_configs.llm_config_sber import get_sber_config
import json
import asyncio
import logging
import sys
from typing import AsyncGenerator, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
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
    "GigaChat-Pro": "GigaChat-Pro"
}

# Initialize Sber configs for both models
try:
    logger.info("Initializing Sber configurations...")
    sber_configs = {}
    for model_id, model_name in MODELS.items():
        try:
            sber_configs[model_id] = get_sber_config(model_name)
            logger.info(f"Successfully initialized config for {model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize config for {model_id}: {str(e)}")
            raise
except Exception as e:
    logger.error(f"Failed to initialize Sber configurations: {str(e)}")
    raise

def convert_to_openai_format(gigachat_response: Dict[str, Any]) -> Dict[str, Any]:
    """Convert GigaChat response to OpenAI format."""
    return {
        "id": f"chatcmpl-{datetime.now().timestamp()}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": gigachat_response.get("model", "GigaChat"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": gigachat_response["choices"][0]["message"]["content"]
                },
                "finish_reason": "stop"
            }
        ],
        "usage": gigachat_response.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })
    }

async def process_stream_response(stream) -> AsyncGenerator[str, None]:
    """Process streaming response in OpenAI format."""
    try:
        for chunk in stream:
            # Convert the chunk to the OpenAI streaming format
            response_data = {
                "id": f"chatcmpl-{datetime.now().timestamp()}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": "GigaChat",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk.choices[0].delta.content if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content') else "",
                            "role": "assistant"
                        },
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(response_data)}\n\n"
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
        logger.info(f"Received chat completion request for model: {body.get('model', 'GigaChat')}")
        
        # Extract messages and model from the request
        messages = body.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Get the requested model, default to GigaChat
        model_id = body.get("model", "GigaChat")
        if model_id not in MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model. Available models: {', '.join(MODELS.keys())}"
            )

        # Get the appropriate config for the requested model
        config = sber_configs[model_id]
        
        # Check if streaming is requested
        stream = body.get("stream", False)
        
        # Create chat completion using GigaChat
        if stream:
            response = config.client.stream(messages[-1]["content"])  # GigaChat only takes the last message for streaming
            return StreamingResponse(
                process_stream_response(response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Transfer-Encoding": "chunked"
                }
            )
        
        # For non-streaming requests
        response = config.client.chat(messages=messages)
        openai_format_response = convert_to_openai_format(response)
        
        logger.info(f"Successfully processed request for {model_id}")
        return openai_format_response

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
            models_data.append({
                "id": model_id,
                "object": "model",
                "created": timestamp,
                "owned_by": "Sber",
                "permission": [],
                "root": model_id,
                "parent": None
            })
        
        return {
            "data": models_data,
            "object": "list"
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Sber OpenAI wrapper service")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Sber OpenAI wrapper service")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
