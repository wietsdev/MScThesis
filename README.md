# MScThesis -- Multilingual Climate Information LLM Benchmark

MSc thesis project building and evaluating a multilingual climate QA benchmark across multiple LLMs.

## What this is

A pipeline that builds a 650-item English benchmark across three task types (MCQ, claim verification, freeform), translates it into multiple languages, and evaluates frontier and open-weight LLMs. The English set is constructed from four public datasets with topic-diversity sampling guided by the Liu et al. 26-topic climate taxonomy.

## Project structure

```
pipeline/                          # numbered stages: run in order
  01_select.py       sample from ClimaQA-Gold, Climate-FEVER, PIRA 2.0
  01b_load_clinb.py  load all 200 CLINB freeform questions from Kaggle
  02_generate_mcq.py generate 100 MCQ from source docs (stub, needs source files)
  03_freeze.py       merge selections -> english_master_v{N}.jsonl + manifest

src/
  item_schema.py   unified Item dataclass; every benchmark item serialises to this
  models.py        generate(prompt, model_id) -> str; single gateway adapter;
                   MODEL_REGISTRY is the only place model strings live;
                   cost + token logging on every real call
  cache.py         SQLite response cache at .cache/responses.db;
                   keyed on SHA-256(model_id + prompt + params)
  scorer.py        score_file() dispatches on split;
                   MCQ: two-pass parser; cloze: normalised match;
                   accuracy_by_field(scores, field) works for any axis

scripts/
  plot_selection.py   2x3 figure: source breakdown, topic pies, Liu coverage
  plot_clinb.py       reproduce CLINB paper Fig 4 (topic x level by WG)
  inspect_data.py     print sample items from ClimaQA-Gold
  run_eval.py         dry-run: one model, N items -> responses.jsonl
  run_matrix.py       English-only matrix runner (ClimaQA MCQ + cloze)
  analyse_matrix.py   per-model accuracy, complexity breakdown, comparison table
  enrich_with_topics.py  join scores with Liu taxonomy -> scores_enriched.jsonl
  cost_summary.py     total spend, remaining budget, per-model token counts

tests/
  test_scorer.py   unit tests for MCQ and cloze parsers

data/                   gitignored (download separately)
  climaqa_topic_taxonomy.jsonl   Liu et al. 26-topic taxonomy
  selections/                    per-source selection files (output of 01_select.py)
  english_master_v{N}.jsonl      frozen master datasets

data_manifest/          committed -- lightweight freeze records
  english_master_v{N}_manifest.json   item counts, SHA-256, source breakdown
  selection_ids.json                  original dataset IDs for reproducibility

figures/
  selection_overview.png   topic/source/label distribution charts
  topics_by_source.png     ClimaQA vs Climate-FEVER topic pies
  clinb_overview.png       CLINB topic x difficulty by working group

results/         gitignored -- eval output (see layout below)
.cache/          gitignored -- responses.db + cost_log.jsonl
.env             gitignored -- gateway credentials
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Fill in `.env` with gateway credentials:

```
GATEWAY_URL=...
GATEWAY_URL_LONG=...
GATEWAY_API_KEY=...
```

Kaggle credentials required for CLINB (`~/.kaggle/kaggle.json`).

## English benchmark -- current state

**Target: 650 items** (550 currently frozen as v4, 100 generated MCQ pending)

| Source | Type | N | Options / Labels | Notes |
|---|---|---|---|---|
| ClimaQA-Gold | MCQ | 100 | 4-option | topic-diverse sample; gold = letter |
| PIRA 2.0 | MCQ | 100 | 5-option | ocean domain; EN/PT anchor for translation |
| Generated | MCQ | 100 | 4-option | targets 15 topic gaps; pending source docs |
| Climate-FEVER | Claim | 150 | SUPPORTS / REFUTES / NEI / DISPUTED | label-stratified + topic-diverse |
| CLINB | Freeform | 200 | n/a | unscored until LLM judge |

MCQ sets are scored separately (different option counts). Freeform items carry `gold=""` until the LLM judge is built.

**Topic coverage (Liu et al. 26-topic taxonomy):**
- 24/26 topics covered across ClimaQA + Climate-FEVER
- Missing: D1 (Agricultural Adaptation), D3 (Public Health Adaptation)
- Thin (<8 items): D2, C4, D5, E5, E3, C5, C6, E4 -- primary targets for generated MCQ

**Building the English master:**
```bash
uv run python pipeline/01_select.py         # ClimaQA + Climate-FEVER + PIRA
uv run python pipeline/01b_load_clinb.py    # CLINB (requires Kaggle credentials)
# uv run python pipeline/02_generate_mcq.py --source-dir data/sources/generation
uv run python pipeline/03_freeze.py         # merges + validates + writes manifest
```

**Reproducing a specific version:** `data_manifest/selection_ids.json` lists the exact dataset IDs used for each source.

## Datasets

**ClimaQA-Gold** -- `Rose-STL-Lab/ClimaQA`, config `Gold` (auto-downloaded).
MCQ: `Question`, `Options` (a-d separated by `--------------------`), `Answer`, `Complexity` (BASE / REASONING / HYPOTHETICAL).

**PIRA 2.0** -- `paulopirozelli/pira`, config `mcqa`, split `test`. CC-BY-4.0.
Fields: `question`, `A`-`E` (options), `alternative` (correct letter), `id`.

**Climate-FEVER** -- `tdiggelm/climate_fever`. License unknown (check paper).
Fields: `claim_id`, `claim`, `claim_label` (0-3), `evidences`. Labels: 0=SUPPORTS, 1=REFUTES, 2=NOT_ENOUGH_INFO, 3=DISPUTED.

**CLINB** -- Kaggle `deepmind/clinb-questions`. CC-BY-4.0.
200 real user questions from ChatClimate.ai. Fields: `question_id`, `question`, `level` (Advanced / High Confidence / Open), `wg` (I/II/III), `topic`. No single gold answer -- evaluated by LLM judge.

**Topic taxonomy** -- `data/climaqa_topic_taxonomy.jsonl`. 26 topics. Covers ClimaQA-Gold (84-88%) and Climate-FEVER (96%) items; join key = `ClimaQA_Gold_<split>_<idx>` or `Climate_FEVER_<claim_id>`.

## Models

Configured in `src/models.py` under `MODEL_REGISTRY`. To add a model: one new dict entry.

| ID | Gateway model string | Notes |
|----|----------------------|-------|
| `claude-haiku` | `anthropic.claude-haiku-4-5-20251001-v1:0` | Frontier |
| `qwen3-32b` | `qwen.qwen3-32b-v1:0` | Open-weight |
| `llama3-8b` | `meta.llama3-8b-instruct-v1:0` | Open-weight |
| `ministral-3-8b` | `mistral.ministral-3-8b-instruct` | Open-weight |

All calls go through the same gateway. `long_endpoint: True` routes to `GATEWAY_URL_LONG`.

## Running existing English evals (ClimaQA only)

```bash
uv run python scripts/run_matrix.py        # all models x MCQ + cloze
uv run python scripts/analyse_matrix.py   # comparison table + breakdowns
uv run python scripts/enrich_with_topics.py          # adds topic labels
uv run python scripts/cost_summary.py     # budget check
uv run pytest tests/
```

## Results layout

```
results/
  matrix_<ts>/
    matrix_config.json        models, splits, seed, item indices
    <model>_<split>/
      config.json             model, dataset, split, n_items
      responses.jsonl         id, question, gold, complexity, prompt, response
      scores.jsonl            id, question, gold, complexity, raw_response, parsed, correct
      misses.jsonl            wrong/unparseable items only
      scores_enriched.jsonl   scores + topics field (after enrich step)
```

Item `id` is the original dataset row index -- stable join key across all models.

## Caching and cost tracking

Every real model call is cached in `.cache/responses.db`. Re-running with the same items costs nothing. Cache key: `SHA-256(model_id + prompt + params)`.

Every real call appends to `.cache/cost_log.jsonl`:
```json
{"ts": "...", "model_id": "...", "cost": 0.0001, "remaining": 19.99, "input_tokens": 124, "output_tokens": 4}
```

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

@article{pirozelli2024pira,
  title   = {Benchmarks for Pir{\'a} 2.0, a Reading Comprehension Dataset about the Ocean,
             the Brazilian Coast, and Climate Change},
  author  = {Paulo Pirozelli and Jo{\~a}o Marcos Mois{\'e}s Filho and Flavio Figueiredo and
             Fabio Cozman and Denis Deratani Mau{\'a}},
  journal = {Data Intelligence},
  volume  = {6},
  number  = {1},
  pages   = {29--55},
  year    = {2024},
  publisher = {MIT Press}
}

@inproceedings{diggelmann2020climatefever,
  title     = {{CLIMATE-FEVER}: A Dataset for Verification of Real-World Climate Claims},
  author    = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and
               Massimiliano Ciaramita and Markus Leippold},
  booktitle = {Tackling Climate Change with Machine Learning Workshop, NeurIPS},
  year      = {2020}
}

@misc{huebscher2025clinb,
  title         = {{CLINB}: A Climate Intelligence Benchmark for Foundational Models},
  author        = {Michelle Chen Huebscher and Katharine Mach and Aleksandar Stani{\'c} and
                   Markus Leippold and Ben Gaiarin and Zeke Hausfather and Elisa Rawat and
                   Erich Fischer and Massimiliano Ciaramita and Joeri Rogelj and
                   Christian Buck and Lierni Sestorain Saralegui and Reto Knutti},
  year          = {2025},
  eprint        = {2511.11597},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2511.11597}
}
```
