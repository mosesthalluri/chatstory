"""
LLM client. Unified interface that talks to either Groq (cloud, free
tier) or Ollama (local, free, slower). Switched via USE_OLLAMA in .env.

Public API:
    response = await llm.complete(messages, model_size="fast")

Both backends speak the OpenAI chat-completions format, so the prompt
construction code never has to know which one is in use.
"""

import asyncio
from typing import Literal

import httpx

from .settings import settings


class LLMError(Exception):
    pass


async def _groq_complete(
    messages: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Call Groq's OpenAI-compatible endpoint."""
    if not settings.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set in .env")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(3):
            try:
                resp = await client.post(url, headers=headers, json=body)
                if resp.status_code == 429:
                    # Rate limited — back off
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if attempt == 2:
                    raise LLMError(f"Groq API error: {e.response.status_code} {e.response.text[:200]}")
                await asyncio.sleep(2 ** attempt)
            except (httpx.RequestError, KeyError) as e:
                if attempt == 2:
                    raise LLMError(f"Groq request failed: {e}")
                await asyncio.sleep(2 ** attempt)

    raise LLMError("Groq completion failed after retries")


async def _ollama_complete(
    messages: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Call local Ollama. Uses Ollama's native chat API.

    CRITICAL: We set num_ctx=16384 explicitly. Ollama's DEFAULT context
    window is only 2048 tokens, which silently truncates the BEGINNING
    of any prompt longer than that. Our chapter prompts are ~2500 tokens,
    so the default would lose the system context and all the message
    history, leaving only the output-format instructions. The LLM would
    then write 'factual summaries' about nothing.

    With 16K context, Llama 3.1 8B and similar models have plenty of
    room for our largest realistic prompt (~6K tokens for chapters with
    full conversation). Llama 3.1 supports up to 128K but quality
    degrades past 32K — 16K is the safe sweet spot.

    If you want to verify Ollama is honoring this: check the model's
    metadata with `ollama show <model>`. The PARAMETER section should
    show `num_ctx 16384` for our requests.
    """
    url = f"{settings.OLLAMA_HOST.rstrip('/')}/api/chat"

    # Estimate prompt size and warn if approaching context limit
    total_chars = sum(len(m.get("content", "")) for m in messages)
    approx_tokens = total_chars // 4  # rough rule: ~4 chars per token
    if approx_tokens > settings.OLLAMA_NUM_CTX * 0.85:
        print(
            f"[llm] WARNING: prompt ~{approx_tokens} tokens, "
            f"context limit is {settings.OLLAMA_NUM_CTX}. "
            f"Output may be poor — increase OLLAMA_NUM_CTX in .env."
        )

    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,            # output token limit
            "num_ctx": settings.OLLAMA_NUM_CTX,   # input context window
        },
    }

    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except httpx.RequestError as e:
            raise LLMError(
                f"Could not reach Ollama at {settings.OLLAMA_HOST}. "
                f"Is `ollama serve` running? Original error: {e}"
            )
        except httpx.HTTPStatusError as e:
            detail = e.response.text.strip()
            suffix = f" Response: {detail[:500]}" if detail else ""
            raise LLMError(
                f"Ollama API error: {e.response.status_code} "
                f"{e.response.reason_phrase}.{suffix}"
            )
        except KeyError as e:
            raise LLMError(f"Ollama returned an unexpected response: missing {e}")


async def complete(
    messages: list[dict],
    model_size: Literal["fast", "strong"] = "fast",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Get a completion from the configured LLM backend.

    `messages` is OpenAI chat format: [{"role": "system", "content": "..."}, ...]
    `model_size` picks between fast (cheap, for bulk) and strong (better, for
    final outputs).
    """
    if settings.USE_OLLAMA:
        model = settings.OLLAMA_MODEL_FAST if model_size == "fast" else settings.OLLAMA_MODEL_STRONG
        return await _ollama_complete(messages, model, temperature, max_tokens)
    else:
        model = settings.GROQ_MODEL_FAST if model_size == "fast" else settings.GROQ_MODEL_STRONG
        return await _groq_complete(messages, model, temperature, max_tokens)
