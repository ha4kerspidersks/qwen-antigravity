# Qwen Antigravity Bridge

High-performance FastAPI bridge connecting Google Antigravity / Gemini-protocol client requests to Qwen3.8-27B hosted on Hugging Face Router or OpenAI-compatible backends.

## Overview
This service translates Google Antigravity IDE and Gemini streaming/non-streaming RPC payloads into OpenAI/Hugging Face Chat Completion requests, supporting:
- Bidirectional SSE streaming response chunk translation.
- System instructions and role remapping (`user` / `model` / `assistant`).
- Tool calls and structured output handling.
- Zero local credential storage (utilizes environment variable `HF_TOKEN`).

## Prerequisites
- Python 3.10+
- `fastapi`, `uvicorn`, `httpx`

## Running Locally
```bash
export HF_TOKEN="your_huggingface_token"
uvicorn bridge:app --host 0.0.0.0 --port 8000
```
