import os

import anthropic
import openai
from dotenv import load_dotenv

import cache

load_dotenv()

# Registry: short model ID -> backend config.
# To add a model: add an entry here, nothing else changes.
# default_params are merged with any params passed to generate(); caller wins on conflicts.
MODEL_REGISTRY = {
    "gpt-4o": {
        "backend": "openai",
        "model_name": "gpt-4o",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
    "claude-haiku-3-5": {
        "backend": "anthropic",
        "model_name": "claude-haiku-4-5-20251001",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
    # Open-weight placeholders -- backend is openai-compatible (Together AI, Ollama, vLLM).
    # Set OPENWEIGHT_BASE_URL and OPENWEIGHT_API_KEY in .env when backend is decided.
    "llama-3.1-8b": {
        "backend": "openai_compatible",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "base_url_env": "OPENWEIGHT_BASE_URL",
        "api_key_env": "OPENWEIGHT_API_KEY",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
    "mistral-7b": {
        "backend": "openai_compatible",
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_url_env": "OPENWEIGHT_BASE_URL",
        "api_key_env": "OPENWEIGHT_API_KEY",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
}


def generate(prompt: str, model_id: str, **params) -> str:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_id}'. Add it to MODEL_REGISTRY in models.py.")

    cfg = MODEL_REGISTRY[model_id]
    effective_params = {**cfg.get("default_params", {}), **params}

    cached = cache.get(model_id, prompt, effective_params)
    if cached is not None:
        return cached

    backend = cfg["backend"]
    if backend == "openai":
        response = _call_openai(prompt, cfg, effective_params)
    elif backend == "anthropic":
        response = _call_anthropic(prompt, cfg, effective_params)
    elif backend == "openai_compatible":
        response = _call_openai_compatible(prompt, cfg, effective_params)
    else:
        raise ValueError(f"Unknown backend: '{backend}'.")

    cache.set(model_id, prompt, effective_params, response)
    return response


def _call_openai(prompt: str, cfg: dict, params: dict) -> str:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=cfg["model_name"],
        messages=[{"role": "user", "content": prompt}],
        **params,
    )
    return resp.choices[0].message.content.strip()


def _call_anthropic(prompt: str, cfg: dict, params: dict) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=cfg["model_name"],
        messages=[{"role": "user", "content": prompt}],
        **params,
    )
    return resp.content[0].text.strip()


def _call_openai_compatible(prompt: str, cfg: dict, params: dict) -> str:
    client = openai.OpenAI(
        api_key=os.environ.get(cfg["api_key_env"], "ollama"),
        base_url=os.environ[cfg["base_url_env"]],
    )
    resp = client.chat.completions.create(
        model=cfg["model_name"],
        messages=[{"role": "user", "content": prompt}],
        **params,
    )
    return resp.choices[0].message.content.strip()
