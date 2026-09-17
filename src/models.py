import json
import os
import time
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
    # Called via gateway for the eval stage; same checkpoint as the local HF
    # model used later for hidden-state/layer investigation (deliberately
    # kept separate -- do not run this locally until that phase begins).
    "gemma3-12b": {
        "model_name": "google.gemma-3-12b-it",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
    "llama3-70b": {
        "model_name": "meta.llama3-70b-instruct-v1:0",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
    "mistral-large": {
        "model_name": "mistral.mistral-large-2402-v1:0",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
    # max_tokens=500 (not 32): confirmed the same invisible-reasoning-token
    # behavior as claude-opus-5, but prompt-dependent here -- 0 empty
    # responses on MCQ items, but 84/150 on climate_fever specifically (the
    # harder evidence-synthesis task), which explains its anomalously low
    # 24.7% climate_fever score in the first full run (see results/
    # 20260825-101153_gpt-oss-20b_full) -- mostly a token-budget artifact,
    # not a real capability gap. Re-run after this fix before trusting that number.
    "gpt-oss-20b": {
        "model_name": "openai.gpt-oss-20b-1:0",
        "default_params": {"max_tokens": 500, "temperature": 0.0},
        "long_endpoint": True,
    },
    "deepseek-v3": {
        "model_name": "deepseek.v3-v1:0",
        "default_params": {"max_tokens": 32, "temperature": 0.0},
    },
    # max_tokens=500 (not 32): confirmed nova-pro reasons verbosely in Chinese
    # before answering on climaqa_gold/pira2 specifically (not generated_mcq/
    # climate_fever, and not other non-English languages tested -- Dutch,
    # Arabic, Bengali, Portuguese all had 0-1 unparseable at max_tokens=32).
    # At 32 the Chinese reasoning gets cut off mid-sentence before ever
    # reaching an answer letter: 16/100 unparseable on climaqa_gold, 13/100
    # on pira2 (see results/*_nova-pro_zh_full). max_tokens is a ceiling, not
    # a target, so this doesn't add cost for languages that already answer
    # short -- only unblocks Chinese. Re-run zh after this lands.
    "nova-pro": {
        "model_name": "amazon.nova-pro-v1:0",
        "default_params": {"max_tokens": 500, "temperature": 0.0},
    },
    # Frontier tier, used for the multilingual "few" (alongside gemma3-12b
    # and gpt-5.6-sol) -- deliberately picked for genuine capability, not cost.
    # max_tokens=500 (not 32): confirmed by direct testing that this reasoning
    # -heavy model spends tokens on invisible internal reasoning before any
    # visible text, even for a plain MCQ -- at max_tokens=32 it used the full
    # budget on reasoning and returned an EMPTY content list (real cost
    # incurred, zero usable output). 500 was enough headroom in testing
    # (~90-100 tokens actually used for a real benchmark item).
    "claude-opus-5": {
        "model_name": "eu.anthropic.claude-opus-5",
        "default_params": {"max_tokens": 800, "temperature": 0.0},
        "long_endpoint": True,
    },
    # "Sol" = OpenAI's peak-flagship tier (slowest/most expensive of
    # sol/terra/luna) -- chosen to match claude-opus-5's tier for a genuine
    # frontier-vs-frontier comparison, not terra/luna's cheaper tiers.
    # Same max_tokens=500 headroom as claude-opus-5, as a preemptive
    # precaution -- not yet directly confirmed for this specific model, but
    # OpenAI's reasoning-tier models are well known for the same
    # invisible-reasoning-token behavior, so the identical failure mode is
    # plausible here too. Revisit if this model produces empty responses too.
    "gpt-5.6-sol": {
        "model_name": "global.openai.gpt-5.6-sol",
        "default_params": {"max_tokens": 800, "temperature": 0.0},
        "long_endpoint": True,
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
    cost_str = f"${cost:.4f}" if cost is not None else "n/a"
    remaining_str = f"${remaining:.2f}" if remaining is not None else "n/a"
    print(f"  [{model_id}] cost={cost_str}  remaining={remaining_str}  tokens={input_tokens}in/{output_tokens}out")
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

    # Retry transient failures (read timeouts, connection resets, 5xx/429
    # responses) with backoff -- long batch runs make hundreds of sequential
    # calls and a single blip otherwise kills the whole run with no cached
    # recovery point for that one call.
    #
    # 429 (rate limit) gets its own, much more patient backoff than other
    # transient errors: a 429 means "you're going too fast", which -- unlike a
    # one-off timeout or 5xx blip -- can take a real amount of time to clear,
    # especially under the kind of sustained multi-process concurrent load
    # this project now runs (many SUT eval runs launched in parallel).
    # Confirmed by an actual incident: running 9 model evals simultaneously
    # exhausted the old 4-attempt/2-4-8s backoff long before the rate limit
    # cleared, killing every run. Respects a Retry-After header if the
    # gateway sends one.
    max_attempts = 4
    max_attempts_429 = 6
    for attempt in range(1, max(max_attempts, max_attempts_429) + 1):
        try:
            # Client-side timeout must exceed the endpoint's own cap, not just
            # match it -- otherwise the client gives up right as the gateway
            # would have answered. Standard endpoint has a ~30s hard cap
            # (AWS API Gateway integration limit); GATEWAY_URL_LONG allows up
            # to 120s.
            resp = requests.post(url, json=body, headers=headers,
                                  timeout=130 if long_endpoint else 60)
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                 requests.exceptions.HTTPError) as e:
            is_429 = isinstance(e, requests.exceptions.HTTPError) and e.response is not None and e.response.status_code == 429
            is_retryable = isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)) or (
                e.response is not None and (e.response.status_code == 429 or e.response.status_code >= 500)
            )
            attempt_limit = max_attempts_429 if is_429 else max_attempts
            if not is_retryable or attempt >= attempt_limit:
                raise
            if is_429:
                retry_after = e.response.headers.get("Retry-After") if e.response is not None else None
                wait = float(retry_after) if retry_after else 10 * (2 ** (attempt - 1))  # 10, 20, 40, 80, 160s
            else:
                wait = 2 ** attempt
            print(f"  [retry {attempt}/{attempt_limit}] {type(e).__name__}"
                  f"{' (429 rate limit)' if is_429 else ''} -- retrying in {wait:.0f}s")
            time.sleep(wait)

    if long_endpoint:
        # GATEWAY_URL_LONG is a separate Lambda with an OpenAI-compatible
        # response shape (choices[0].message.content), NOT the standard
        # gateway's {"content": ...} shape -- confirmed by a direct test call
        # (bug found 2026-08-31: 9 claude-opus-5 matrix runs crashed instantly
        # with KeyError('content') the moment long_endpoint was first turned
        # on, because this branch didn't exist yet). It also doesn't return
        # cost/remaining-budget fields at all, so those are genuinely
        # unavailable here -- reported as None (not 0.0, which would silently
        # corrupt spend totals in cost_log.jsonl by looking like a free call)
        # rather than guessed at.
        text = (data["choices"][0]["message"]["content"] or "").strip()
        cost = None
        remaining = None
        input_tokens = data["usage"]["prompt_tokens"]
        output_tokens = data["usage"]["completion_tokens"]
        return text, cost, remaining, input_tokens, output_tokens

    # frontier models return content as a list of dicts; open-weight as a plain string
    raw_content = data["content"]
    if isinstance(raw_content, str):
        text = raw_content.strip()
    elif raw_content:
        text = raw_content[0]["text"].strip()
    else:
        # Empty content list: seen from reasoning-heavy models (claude-opus-5,
        # gpt-oss-20b) that spend the ENTIRE max_tokens budget on invisible
        # internal reasoning before any visible answer -- real cost is still
        # incurred (outputTokens == max_tokens) but there's no usable text.
        # Bumping max_tokens reduces how often this happens but doesn't
        # eliminate it (confirmed: still recurred at max_tokens=500 on some
        # items). Return "" (scored as unparseable downstream) rather than
        # crash the whole run over one bad response -- losing every
        # already-completed item's real spend to one edge case is worse than
        # one item scoring unparseable.
        text = ""
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
