# MScThesis -- Climate Information Integrity LLM Evaluation Harness

MSc thesis project evaluating climate information integrity in large language models.

## What this is

An evaluation harness that runs climate questions through several LLMs and scores the answers. Phase 1 uses ClimaQA-Gold (multiple-choice and cloze questions) with exact-match scoring.

## Project structure

```
data/            downloaded datasets (gitignored)
scripts/         runnable scripts
  inspect_data.py    load and print sample dataset items
  run_eval.py        run questions through a model, save responses (coming next)
src/             importable harness modules
  models.py          uniform generate(prompt, model_id) interface + model registry
  cache.py           SQLite response cache, keyed on (model, prompt, params)
  prompts.py         prompt builders for MCQ and cloze (coming next)
  scorer.py          scoring logic, isolated from model calling (coming next)
results/         eval output files (gitignored)
.cache/          response cache database (gitignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv sync
```

Copy `.env` and fill in your API keys:

```bash
cp .env .env  # .env is already present -- just open it and replace the placeholder values
```

Key(s) needed:

- `OPENAI_API_KEY` -- for GPT-4o
- `ANTHROPIC_API_KEY` -- for Claude Haiku
- `OPENWEIGHT_BASE_URL` / `OPENWEIGHT_API_KEY` -- for open-weight models (backend TBD)

## Models

Configured in `src/models.py` under `MODEL_REGISTRY`. Adding a model is a config change only -no code changes elsewhere.

| ID | Backend | Model |
|----|---------|-------|
| `gpt-4o` | OpenAI | gpt-4o |
| `claude-haiku-3-5` | Anthropic | claude-haiku-4-5-20251001 |
| `llama-3.1-8b` | OpenAI-compatible | meta-llama/Llama-3.1-8B-Instruct-Turbo |
| `mistral-7b` | OpenAI-compatible | mistralai/Mistral-7B-Instruct-v0.3 |

## Usage

Inspect the ClimaQA-Gold dataset:

```bash
uv run python scripts/inspect_data.py
```

## Dataset

ClimaQA-Gold via Hugging Face: `Rose-STL-Lab/ClimaQA`, config `Gold`.

| Split | Count | Used in Phase 1 |
|-------|-------|-----------------|
| MCQ | 292 | yes |
| Cloze | 160 | yes |
| FFQ | 181 | no |

MCQ fields: `Question`, `Options` (a-d), `Answer` (single letter), `Complexity` (BASE / REASONING / HYPOTHETICAL).
Cloze fields: `Question` (contains `<MASK>`), `Answer` (word or short phrase).

## Results layout

Each run produces a folder under `results/`:

```
results/
  20260710-143022_gpt-4o_climaqa_mcq/
    config.json       model, dataset, split, timestamp, generation params
    responses.jsonl   one line per item: id, prompt, raw response
    scores.jsonl      one line per item: id, predicted, gold, correct, complexity
    misses.jsonl      wrong items only, for manual inspection
```
