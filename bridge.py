import os
import json
import httpx

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


app = FastAPI(title="Qwen3.8-27B Antigravity Bridge")

HF_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "Qwen/Qwen3.8-27B"


# ============================================================
# AUTH
# ============================================================

def get_headers():
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set")

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# ============================================================
# GEMINI -> OPENAI
# ============================================================

def gemini_to_openai(body):
    contents = body.get("contents", [])

    messages = []

    system_instruction = body.get("systemInstruction")

    if system_instruction:
        parts = system_instruction.get("parts", [])

        text = "".join(
            p.get("text", "")
            for p in parts
            if isinstance(p, dict)
        )

        if text:
            messages.append({
                "role": "system",
                "content": text,
            })

    for item in contents:

        role = item.get("role", "user")

        if role == "model":
            role = "assistant"

        parts = item.get("parts", [])

        text_parts = []

        for part in parts:

            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])

        if text_parts:
            messages.append({
                "role": role,
                "content": "\n".join(text_parts),
            })

    result = {
        "model": MODEL,
        "messages": messages,
    }

    generation_config = body.get(
        "generationConfig",
        {}
    )

    if "temperature" in generation_config:
        result["temperature"] = generation_config["temperature"]

    if "maxOutputTokens" in generation_config:
        result["max_tokens"] = generation_config["maxOutputTokens"]

    if "topP" in generation_config:
        result["top_p"] = generation_config["topP"]

    if "topK" in generation_config:
        result["top_k"] = generation_config["topK"]

    # ========================================================
    # TOOLS
    # ========================================================

    tools = body.get("tools")

    if tools:

        converted_tools = []

        for tool_group in tools:

            if not isinstance(tool_group, dict):
                continue

            declarations = tool_group.get(
                "functionDeclarations",
                []
            )

            for fn in declarations:

                converted_tools.append({
                    "type": "function",
                    "function": {
                        "name": fn.get("name"),
                        "description": fn.get(
                            "description",
                            ""
                        ),
                        "parameters": fn.get(
                            "parameters",
                            {
                                "type": "object",
                                "properties": {},
                            },
                        ),
                    },
                })

        if converted_tools:
            result["tools"] = converted_tools

    return result


# ============================================================
# OPENAI -> GEMINI
# ============================================================

def openai_to_gemini(response):

    choices = response.get("choices", [])

    if not choices:

        return {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "text": ""
                    }],
                },
                "finishReason": "STOP",
            }]
        }

    choice = choices[0]

    message = choice.get(
        "message",
        {}
    )

    parts = []

    content = message.get("content")

    if content:
        parts.append({
            "text": content
        })

    # ========================================================
    # TOOL CALLS
    # ========================================================

    tool_calls = message.get(
        "tool_calls",
        []
    )

    for call in tool_calls:

        function = call.get(
            "function",
            {}
        )

        arguments = function.get(
            "arguments",
            "{}"
        )

        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}

        parts.append({
            "functionCall": {
                "name": function.get(
                    "name",
                    ""
                ),
                "args": arguments,
            }
        })

    finish_reason = choice.get(
        "finish_reason",
        "stop"
    )

    finish_map = {
        "stop": "STOP",
        "length": "MAX_TOKENS",
        "tool_calls": "STOP",
    }

    finish_reason = finish_map.get(
        finish_reason,
        "STOP"
    )

    result = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": parts,
            },
            "finishReason": finish_reason,
        }]
    }

    usage = response.get("usage")

    if usage:

        result["usageMetadata"] = {
            "promptTokenCount": usage.get(
                "prompt_tokens",
                0
            ),
            "candidatesTokenCount": usage.get(
                "completion_tokens",
                0
            ),
            "totalTokenCount": usage.get(
                "total_tokens",
                0
            ),
        }

    return result


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "ok",
        "service": "Qwen3.8-27B Antigravity Bridge",
        "model": MODEL,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model": MODEL,
    }


# ============================================================
# NORMAL GENERATE CONTENT
# ============================================================

@app.post(
    "/v1beta/models/{model}:generateContent"
)
async def generate_content(
    model: str,
    request: Request
):

    body = await request.json()

    try:

        payload = gemini_to_openai(body)

        payload["stream"] = False

        async with httpx.AsyncClient(
            timeout=300
        ) as client:

            response = await client.post(
                HF_URL,
                headers=get_headers(),
                json=payload,
            )

        if response.status_code >= 400:

            try:
                error = response.json()
            except Exception:
                error = {
                    "error": {
                        "message": response.text,
                        "type": "huggingface_error",
                    }
                }

            return JSONResponse(
                status_code=response.status_code,
                content=error,
            )

        return JSONResponse(
            content=openai_to_gemini(
                response.json()
            )
        )

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": "bridge_error",
                }
            },
        )


# ============================================================
# STREAM GENERATE CONTENT
#
# IMPORTANT:
# This endpoint deliberately uses Response()
# instead of StreamingResponse().
# ============================================================

@app.post(
    "/v1beta/models/{model}:streamGenerateContent"
)
async def stream_generate_content(
    model: str,
    request: Request
):

    body = await request.json()

    try:

        payload = gemini_to_openai(body)

        # HF request itself is non-streaming.
        payload["stream"] = False

        async with httpx.AsyncClient(
            timeout=300
        ) as client:

            response = await client.post(
                HF_URL,
                headers=get_headers(),
                json=payload,
            )

        if response.status_code >= 400:

            try:
                error_data = response.json()
            except Exception:
                error_data = {
                    "error": {
                        "message": response.text,
                        "type": "huggingface_error",
                    }
                }

            sse_body = (
                "data: "
                + json.dumps(
                    error_data,
                    separators=(",", ":")
                )
                + "\n\n"
                + "data: [DONE]\n\n"
            )

            return Response(
                content=sse_body,
                status_code=response.status_code,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Qwen-Bridge": "SSE",
                },
            )

        gemini_response = openai_to_gemini(
            response.json()
        )

        # ====================================================
        # BUILD REAL SSE BODY
        # ====================================================

        sse_body = (
            "data: "
            + json.dumps(
                gemini_response,
                separators=(",", ":")
            )
            + "\n\n"
            + "data: [DONE]\n\n"
        )

        return Response(
            content=sse_body,
            status_code=200,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Qwen-Bridge": "SSE",
            },
        )

    except Exception as exc:

        error_data = {
            "error": {
                "message": str(exc),
                "type": "bridge_error",
            }
        }

        sse_body = (
            "data: "
            + json.dumps(
                error_data,
                separators=(",", ":")
            )
            + "\n\n"
            + "data: [DONE]\n\n"
        )

        return Response(
            content=sse_body,
            status_code=500,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Qwen-Bridge": "SSE",
            },
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8787,
    )
