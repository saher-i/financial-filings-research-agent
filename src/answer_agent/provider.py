"""Fireworks chat completion adapter; no secret enters graph state or responses."""

import os

DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"


class FireworksChatProvider:
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        key = os.getenv("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError("Set FIREWORKS_API_KEY in .env or the server environment")
        self.model = model or os.getenv("FIREWORKS_CHAT_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=key, base_url="https://api.fireworks.ai/inference/v1",
                             timeout=45.0, max_retries=1)

    def complete(self, messages: list[dict], tools: list[dict], *, max_tokens: int | None = None) -> dict:
        if max_tokens is None:
            max_tokens = 3000 if any(name in self.model.lower() for name in ("deepseek", "glm-5p3")) else 2000
        options = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            options.update({"tools": tools, "tool_choice": "auto"})
        response = self.client.chat.completions.create(**options)
        message = response.choices[0].message
        calls = [{"id": call.id, "type": "function",
                  "function": {"name": call.function.name, "arguments": call.function.arguments}}
                 for call in (message.tool_calls or [])]
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        diagnostics = {"finish_reason": response.choices[0].finish_reason,
                       "prompt_tokens": getattr(usage, "prompt_tokens", None),
                       "completion_tokens": getattr(usage, "completion_tokens", None),
                       "reasoning_tokens": getattr(details, "reasoning_tokens", None)}
        return {"role": "assistant", "content": message.content or "", "tool_calls": calls,
                "diagnostics": diagnostics}
