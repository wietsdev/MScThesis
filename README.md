# MScThesis -- Climate Information Integrity LLM Evaluation Harness

MSc thesis project evaluating climate information integrity in large language models.

## What this is

An evaluation harness that runs climate questions through several LLMs and scores the answers. Phase 1 uses ClimaQA-Gold (multiple-choice and cloze questions) with exact-match scoring.

## Project structure

```
data/        downloaded datasets (gitignored)
scripts/     runnable scripts (download, inspect, later: run eval)
src/         importable harness modules (populated in later phases)
results/     eval output files (gitignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv sync
```

## Usage

Inspect the ClimaQA-Gold dataset:

```bash
uv run python scripts/inspect_data.py
```

## Dataset

ClimaQA-Gold via Hugging Face: `Rose-STL-Lab/ClimaQA`, config `Gold`.
- 292 multiple-choice questions
- 160 cloze questions
- 181 free-form questions (not used in Phase 1)