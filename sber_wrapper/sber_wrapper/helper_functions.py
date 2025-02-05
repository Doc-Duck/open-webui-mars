import json
import asyncio
import re
import base64
from typing import AsyncGenerator, Optional, Dict, Any

from fastapi import logger

from .llm_configs.llm_config_sber import download_image_async


async def process_stream_response(stream) -> AsyncGenerator[str, None]:
    """Process streaming response from OpenAI API."""
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


async def generate_and_get_image(
    client: Any, prompt: str, size: str = "1024x1024", style: Optional[str] = None
) -> Dict[str, Any]:
    """Generate an image using Sber API and return in OpenAI format."""
    try:
        # Prepare system message for style if provided
        messages = []
        if style:
            messages.append({"role": "system", "content": style})

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        # Request image generation
        response = await client.chat.completions.create(
            model="GigaChat-Max", messages=messages, function_call="auto"
        )

        # Extract image ID from response
        content = response.choices[0].message.content
        image_id_match = re.search(r'<img src="([^"]+)"', content)

        if not image_id_match:
            raise ValueError("No image ID found in response")

        image_id = image_id_match.group(1)

        image_bytes = await download_image_async(image_id, client.api_key)

        # Return in OpenAI format
        return {
            "created": int(asyncio.get_event_loop().time()),
            "data": [
                {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}",
                    "b64_json": base64.b64encode(image_bytes).decode("utf-8"),
                }
            ],
        }

    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise
