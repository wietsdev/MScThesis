# MScThesis -- Climate Information Integrity LLM Evaluation Harness

MSc thesis project evaluating climate information integrity in large language models.

## What this is

An evaluation harness that runs climate questions through several LLMs and scores the answers. Phase 1 uses ClimaQA-Gold (MCQ and cloze questions) with exact-match scoring, analysed by complexity tier and topic.

## Project structure

```
data/
  climaqa_topic_taxonomy.jsonl   26-topic taxonomy for 44k climate questions;
                                 83-88% of ClimaQA-Gold items have a direct match

scripts/
  inspect_data.py        print sample MCQ and cloze items from ClimaQA-Gold
  run_eval.py            small dry-run: one model, one split, N items -> responses.jsonl
  run_matrix.py          main runner: all models x both splits x N items, scored
  analyse_matrix.py      per-model accuracy, complexity breakdown, comparison table
  cost_summary.py        total spend, remaining budget, per-model token counts
  enrich_with_topics.py  join scores.jsonl with topic labels -> scores_enriched.jsonl

src/
  models.py    generate(prompt, model_id, **params) -> str
               single gateway adapter; MODEL_REGISTRY is the only place model
               strings live; cost + token logging on every real call
  cache.py     SQLite response cache at .cache/responses.db;
               keyed on SHA-256(model_id + prompt + params)
  scorer.py    score_file(responses_path) dispatches on split;
               MCQ: two-pass parser (exact match then regex fallback);
               cloze: normalised match; miss log for every wrong item;
               accuracy_by_field(scores, field) works for any axis
  prompts.py   stub -- prompt formatting is inline in scripts for now

tests/
  test_scorer.py   9 unit tests for the MCQ parser (no model calls)

results/         gitignored -- see layout below
.cache/          gitignored -- responses.db + cost_log.jsonl
.env             gitignored -- gateway credentials
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv sync
```

Fill in `.env` with gateway credentials (file already exists, replace placeholders):

```
GATEWAY_URL=...
GATEWAY_URL_LONG=...
GATEWAY_API_KEY=...
```

## Models

Configured in `src/models.py` under `MODEL_REGISTRY`. To add a model: one new dict entry, nothing else changes.

| ID | Gateway model string | Notes |
|----|----------------------|-------|
| `claude-haiku` | `anthropic.claude-haiku-4-5-20251001-v1:0` | Frontier, cheapest |
| `qwen3-32b` | `qwen.qwen3-32b-v1:0` | Open-weight, large |
| `llama3-8b` | `meta.llama3-8b-instruct-v1:0` | Open-weight, small |
| `ministral-3-8b` | `mistral.ministral-3-8b-instruct` | Open-weight, small |

All models go through the same gateway endpoint. The `long_endpoint: True` flag in a registry entry routes to `GATEWAY_URL_LONG` for slow/reasoning models.

## Running an eval

**Small dry run (one model, one split):**
```bash
uv run python scripts/run_eval.py          # edit MODEL_ID and SPLIT at top of file
uv run python src/scorer.py results/<folder>/responses.jsonl
```

**Full matrix (all models, both splits, 20 items each):**
```bash
uv run python scripts/run_matrix.py        # samples, calls, scores in one go
uv run python scripts/analyse_matrix.py   # print comparison table + breakdowns
```

**Enrich with topic labels then re-analyse:**
```bash
uv run python scripts/enrich_with_topics.py          # writes scores_enriched.jsonl
```

**Check budget:**
```bash
uv run python scripts/cost_summary.py
```

**Run tests:**
```bash
uv run pytest tests/
```

## Datasets

**ClimaQA-Gold** -- `Rose-STL-Lab/ClimaQA`, config `Gold` (auto-downloaded by scripts).

| Split | Count | Phase 1 |
|-------|-------|---------|
| MCQ | 292 | yes |
| Cloze | 160 | yes |
| FFQ | 181 | no |

MCQ fields: `Question`, `Options` (a-d), `Answer` (single letter), `Complexity` (BASE / REASONING / HYPOTHETICAL).
Cloze fields: `Question` (contains `<MASK>`), `Answer` (word or short phrase).

> Manivannan et al. (2025). ClimaQA: An Automated Evaluation Framework for Climate Question Answering Models. ICLR 2025. https://openreview.net/forum?id=goFpCuJalN

**Topic taxonomy** -- `data/climaqa_topic_taxonomy.jsonl` (copy from HuggingFace, not versioned).
44,917 records from `Westing/LLM-Misalign-Climate-Change`. 26 topics. 83-88% of ClimaQA-Gold items matched by ID (`ClimaQA_Gold_<split>_<row_index>`).

> Liu et al. (2026). LLM Benchmark-User Need Misalignment for Climate Change. arXiv:2603.26106. https://arxiv.org/abs/2603.26106

## References

```bibtex
@inproceedings{manivannan2025climaqa,
  title     = {ClimaQA: An Automated Evaluation Framework for Climate Question Answering Models},
  author    = {Veeramakali Vignesh Manivannan and Yasaman Jafari and Srikar Eranky and
               Spencer Ho and Rose Yu and Duncan Watson-Parris and Yian Ma and
               Leon Bergen and Taylor Berg-Kirkpatrick},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=goFpCuJalN}
}

@misc{liu2026llmbenchmarkuserneedmisalignment,
  title         = {LLM Benchmark-User Need Misalignment for Climate Change},
  author        = {Oucheng Liu and Lexing Xie and Jing Jiang},
  year          = {2026},
  eprint        = {2603.26106},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2603.26106}
}
```

## Results layout

```
results/
  matrix_<ts>/
    matrix_config.json        models, splits, seed, exact item indices
    claude-haiku_mcq/
      config.json             model, dataset, split, n_items
      responses.jsonl         id, question, gold, complexity, prompt, response
      scores.jsonl            id, question, gold, complexity, raw_response, parsed, correct
      misses.jsonl            wrong/unparseable items only
      scores_enriched.jsonl   scores.jsonl + topics field (after enrich step)
    claude-haiku_cloze/  ...
    qwen3-32b_mcq/       ...  (8 subfolders total: 4 models x 2 splits)
```

Item `id` is the original dataset row index -- the join key across all models. `id=57` in `claude-haiku_mcq/` is the same question as `id=57` in `qwen3-32b_mcq/`.

## Caching

Every real model call is cached in `.cache/responses.db`. Re-running `run_matrix.py` with the same items and model costs nothing. The cache key is `SHA-256(model_id + prompt + params)` -- changing any of these busts the cache for that call.

## Cost tracking

Every real call appends one line to `.cache/cost_log.jsonl`:
```json
{"ts": "...", "model_id": "...", "cost": 0.0001, "remaining": 19.99, "input_tokens": 124, "output_tokens": 4}
```
Run `scripts/cost_summary.py` at any time to see total spend and remaining budget.
