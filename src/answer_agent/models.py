"""Discover Fireworks serverless chat models that advertise tool support."""

from __future__ import annotations

import json
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .provider import DEFAULT_MODEL


MODEL_PATTERN = re.compile(r"accounts/[A-Za-z0-9_-]+/models/[A-Za-z0-9_.-]+\Z")
CATALOG_URL = "https://api.fireworks.ai/v1/accounts/fireworks/models"
AFFORDABLE_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
GPT_OSS_MODEL = "accounts/fireworks/models/gpt-oss-120b"
QUALITY_CANDIDATE_MODEL = "accounts/fireworks/models/glm-5p3"


def selectable_models(default: str, discovered: list[dict] | None = None) -> list[dict]:
    """Keep tested and premium tool-capable choices visible before catalog discovery."""
    choices = {default: {"id": default,
                         "label": ("DeepSeek V4.1 Flash · default" if default == DEFAULT_MODEL
                                   else default.rsplit("/", 1)[-1] + " · default")}}
    choices.setdefault(AFFORDABLE_MODEL, {"id": AFFORDABLE_MODEL,
                       "label": "Nemotron Lightning 3.5 30B · lower cost"})
    choices.setdefault(GPT_OSS_MODEL, {"id": GPT_OSS_MODEL,
                       "label": "GPT OSS 120B · alternative"})
    choices.setdefault(QUALITY_CANDIDATE_MODEL, {"id": QUALITY_CANDIDATE_MODEL,
                       "label": "GLM-5.3 · premium quality candidate, higher cost"})
    for model in discovered or []:
        if isinstance(model, dict) and isinstance(model.get("id"), str):
            choices.setdefault(model["id"], model)
    return list(choices.values())


def validate_model(model: str | None) -> str:
    selected = model or DEFAULT_MODEL
    if not isinstance(selected, str) or not MODEL_PATTERN.fullmatch(selected):
        raise ValueError("Choose a valid Fireworks chat model")
    return selected


def discover_models(key: str, fetch=None) -> list[dict]:
    """Read provider metadata only; this does not make an inference request."""
    if not key:
        raise ValueError("Set the Fireworks API key first")
    fetch = fetch or urlopen
    models: dict[str, dict] = {}
    token = ""
    seen = set()
    for _ in range(20):
        params = {"filter": "supports_serverless=true", "pageSize": 200}
        if token:
            params["pageToken"] = token
        request = Request(CATALOG_URL + "?" + urlencode(params),
                          headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
        with fetch(request, timeout=15) as response:
            data = json.load(response)
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError("Fireworks returned an invalid model catalog")
        for entry in data["models"]:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if (isinstance(name, str) and MODEL_PATTERN.fullmatch(name)
                    and entry.get("supportsServerless") is True
                    and entry.get("supportsTools") is True):
                models[name] = {"id": name, "label": entry.get("displayName") or name.rsplit("/", 1)[-1],
                                "context_length": entry.get("contextLength")}
        token = data.get("nextPageToken") or ""
        if not token:
            break
        if token in seen:
            raise ValueError("Fireworks repeated a model catalog page")
        seen.add(token)
    else:
        raise ValueError("Fireworks model catalog exceeded the page limit")
    return sorted(models.values(), key=lambda item: str(item["label"]).lower())
