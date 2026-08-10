import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

import cache

load_dotenv()

COST_LOG = Path(__file__).parent.parent / ".cache" / "cost_log.jsonl"

# Registry: short model ID -> gateway model string + default params.
# To add a model: add one entry here.
# long_endpoint: True routes to GATEWAY_URL_LONG (for slow/reasoning models).
MODEL_REGISTRY = {
    # --- SUT panel (evaluated models) ---
    "claude-haiku": {
        "model_name": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
    "qwen3-32b": {
        "model_name": "qwen.qwen3-32b-v1:0",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
    "llama3-8b": {
        "model_name": "meta.llama3-8b-instruct-v1:0",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
    "ministral-3-8b": {
        "model_name": "mistral.ministral-3-8b-instruct",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
    # --- Pipeline utility models (NOT SUTs — do not evaluate these) ---
    # Generator: different family from most SUTs; same model used for translation
    "qwen3-235b-generator": {
        "model_name": "qwen.qwen3-235b-a22b-2507-v1:0",
        "default_params": {"max_tokens": 800, "temperature": 0.5},
    },
    # Verifier: different family from generator to avoid shared blind spots
    "claude-sonnet": {
        "model_name": "eu.anthropic.claude-sonnet-4-6",
        "default_params": {"max_tokens": 256, "temperature": 0.0},
    },
}


def generate(prompt: str, model_id: str, **params) -> str:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_id}'. Add it to MODEL_REGISTRY in src/models.py."
        )

    cfg = MODEL_REGISTRY[model_id]
    effective_params = {**cfg.get("default_params", {}), **params}

    cached = cache.get(model_id, prompt, effective_params)
    if cached is not None:
        return cached

    text, cost, remaining, input_tokens, output_tokens = _call_gateway(
        model_name=cfg["model_name"],
        prompt=prompt,
        params=effective_params,
        long_endpoint=cfg.get("long_endpoint", False),
    )
    print(f"  [{model_id}] cost=${cost:.4f}  remaining=${remaining:.2f}  tokens={input_tokens}in/{output_tokens}out")
    _log_cost(model_id, cfg["model_name"], cost, remaining, input_tokens, output_tokens)

    cache.set(model_id, prompt, effective_params, text)
    return text


def _call_gateway(
    model_name: str,
    prompt: str,
    params: dict,
    long_endpoint: bool = False,
) -> tuple[str, float, float, int, int]:
    url_var = "GATEWAY_URL_LONG" if long_endpoint else "GATEWAY_URL"
    url = os.environ[url_var]
    api_key = os.environ["GATEWAY_API_KEY"]

    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        **params,
    }
    headers = {
        "X-Api-Key": api_key,
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=body, headers=headers, timeout=35)
    resp.raise_for_status()
    data = resp.json()

    # frontier models return content as a list of dicts; open-weight as a plain string
    raw_content = data["content"]
    if isinstance(raw_content, str):
        text = raw_content.strip()
    else:
        text = raw_content[0]["text"].strip()
    cost = data["usage"]["cost"]
    remaining = data["metadata"]["remaining_quota"]["remaining_budget"]
    input_tokens = data["usage"]["inputTokens"]
    output_tokens = data["usage"]["outputTokens"]
    return text, cost, remaining, input_tokens, output_tokens


def _log_cost(
    model_id: str,
    model_name: str,
    cost: float,
    remaining: float,
    input_tokens: int,
    output_tokens: int,
) -> None:
    COST_LOG.parent.mkdir(exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model_name": model_name,
        "cost": cost,
        "remaining": remaining,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    with COST_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
