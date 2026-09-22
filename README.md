# MScThesis -- Multilingual Climate Information LLM Benchmark

MSc thesis project building and evaluating a multilingual climate QA benchmark across multiple LLMs.

## What this is

A pipeline that builds a 652-item English benchmark across three task types (MCQ, claim
verification, freeform), translates it into 12 languages (4 fully evaluated, 8 translation-only
so far), and evaluates a panel of frontier and open-weight LLMs -- including MCQ
position-bias debiasing, and reference-free/reference-based
translation quality estimation. The English set is constructed from five public/generated
sources with topic-diversity sampling guided by the Liu et al. 26-topic climate taxonomy.

## Project structure

```
pipeline/                              # numbered stages: run roughly in order
  01_select.py         topic-stratified sample from ClimaQA-Gold, Climate-FEVER, PIRA 2.0
  01b_load_clinb.py    load all 200 CLINB freeform questions from Kaggle
  02a_chunk_pdfs.py    extract + chunk + keyword-tag IPCC AR6 report PDFs (for MCQ generation)
  02b_prefilter_chunks.py   cheap LLM screen: does this chunk contain a testable fact?
  02_generate_mcq.py   v1 MCQ generation from IPCC chunks (superseded by v2, kept as fallback)
  02_generate_mcq_v2.py  v2: reframed prompt + 5-gate independent verifier + gap-driven
                         topic allocation; the active generation pipeline
  03_freeze.py         merge selections -> english_master_v{N}.jsonl + manifest;
                       shuffles generated_mcq option order (fixes generation-time position bias)
  04_translate_qwen.py translate the master into a target language via Qwen3-235B (gateway),
                       whole-item JSON calls; cross-check translator
  05_tag_topics.py     backfill liu_topic (unified taxonomy) onto PIRA/CLINB, which don't
                       carry it natively, use liu_topic (not topic) for cross-source analysis
  05_tq_score.py       CometKiwi (reference-free) translation quality scoring, per translator
  05b_tq_score_pira_reference.py   COMET (reference-based) validation vs PIRA's human PT translations
  05c_build_hybrid_translation.py  per-item QE-selection between NLLB and Qwen -> hybrid set
  06_run_eval.py       run one SUT model over a master/hybrid file; --debias-mcq for
                       cyclic-permutation position-bias debiasing

cluster/                                # Myriad HPC (NLLB-200 3.3B, local GPU inference)
  download_nllb.py                     one-time model download to $HOME/Scratch/hf_cache
  translate_pilot.py                   the actual translation script (despite the name --
                                        used for pilot AND full runs, per-sentence-fragment
                                        batched translation via BCP47_BY_FLORES language map)
  translate_full_nllb33b_{pt,hi,sw}.job   core-language full runs
  translate_full_nllb33b_extension.job    8 extension languages, one job, sequential
  translate_topup_*.job, topup_*_ids.txt  targeted re-translation of known-bad items

src/
  item_schema.py   unified Item dataclass; every benchmark item (any source/type/language)
                   serialises to this. topic = native per-source taxonomy; liu_topic =
                   unified taxonomy (backfilled), use liu_topic for cross-source topic analysis
  models.py        generate(prompt, model_id, **params) -> str; single gateway adapter;
                   MODEL_REGISTRY is the only place model strings live (SUT panel vs
                   pipeline-utility models kept in explicitly separate sections); 429-aware
                   patient retry backoff; cost + token logging on every real call
  cache.py         SQLite response cache at .cache/responses.db;
                   keyed on SHA-256(model_id + prompt + params)
  scorer.py        parse/score functions per item_type (mcq/claim/cloze) + the master-schema
                   dispatcher (score_master_record, summarise_by_source); MCQ cyclic-rotation
                   helpers (rotate_mcq_options, aggregate_circular_mcq) for position debiasing

scripts/
  qc_sample.py               interactive terminal QC review of generated MCQ candidates
  extract_accepted.py        turn qc_sample.py's accept/reject log into accepted_v{1,2}.jsonl
  backfill_evidence.py       one-off: patch the `evidence` field into already-translated
                             Climate-FEVER items (both translator scripts originally dropped it)
  sample_judge_review.py     stratified human-validation sample from a completed judge run
  score_judge_agreement.py   judge-vs-human Pearson r / MAD / exact-match, per rubric dimension
  cost_summary.py            total spend, remaining budget, per-model token counts
  plot_selection.py, plot_clinb.py, plot_generation.py, analyse_matrix.py, inspect_data.py,
  run_eval.py, run_matrix.py, enrich_with_topics.py, patch_pira_longtext.py,
  remap_generated_mcq_translation.py, backfill_liu_topic.py   earlier-stage / one-off tooling

tests/
  test_scorer.py   unit tests for MCQ and cloze parsers

data/                   gitignored (download/regenerate separately)
  climaqa_topic_taxonomy.jsonl   Liu et al. 26-topic taxonomy
  selections/                    per-source selection files (output of 01_select.py etc.)
  english_master_v{N}.jsonl      frozen master datasets (v9 current, 652 items)
  full_<lang>_nllb.jsonl, full_<lang>_qwen.jsonl   per-translator translated sets
  full_<lang>_hybrid.jsonl       QE-selected hybrid translation (what eval actually reads)

data_manifest/          committed -- lightweight freeze records
  english_master_v{N}_manifest.json   item counts, SHA-256, source/type breakdown

figures/                selection/generation/CLINB overview charts (see scripts/plot_*.py)

results/                gitignored -- eval + translation-quality-score output (see layout below)
  tq_scores/            CometKiwi/COMET summaries + hybrid-selection summaries, per language
.cache/                 gitignored -- responses.db + cost_log.jsonl
.venv-comet/            separate venv for pipeline/05_tq_score.py -- unbabel-comet needs
                        transformers<5.0, which conflicts with the main env (needs >=5.15.0)
.env                    gitignored -- gateway credentials
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

For translation-quality scoring (`pipeline/05_tq_score.py`, `05b_tq_score_pira_reference.py`),
set up the separate CometKiwi/COMET environment (`.venv-comet`) and accept the gated model
license at huggingface.co/Unbabel/wmt22-cometkiwi-da (`export HF_TOKEN=...`). Run those two
scripts with `.venv-comet/bin/python`, not the main `uv run`.

For NLLB translation, translation runs on UCL Myriad HPC (`cluster/*.job`, SGE scheduler) --
requires the model pre-downloaded to `$HOME/Scratch/hf_cache` via `cluster/download_nllb.py`
(compute nodes have no internet access).

## English benchmark -- current state

**652 items, frozen as v9.**

| Source | Type | N | Options / Labels | Notes |
|---|---|---|---|---|
| ClimaQA-Gold | MCQ | 100 | 4-option | topic-diverse sample; gold = letter |
| PIRA 2.0 | MCQ | 100 | 5-option | ocean domain; no native topic tags (liu_topic backfilled) |
| Generated MCQ | MCQ | 102 | 4-option | IPCC AR6-derived, v2 pipeline: reframed generation
prompt + 5-gate independent verifier + gap-driven topic allocation; human-QC'd before freeze |
| Climate-FEVER | Claim | 150 | SUPPORTS / REFUTES / NOT_ENOUGH_INFO / DISPUTED | label-flattened + topic-diverse; evidence-grounded (Wikipedia sentences kept, untranslated, alongside every translated claim) |
| CLINB | Freeform | 200 | n/a | (eventually not) scored by LLM judge (`pipeline/07_judge_freeform.py`), not exact-match |

MCQ sets are always scored separately by option count (4- vs 5-option), and every source is
reported separately, never blended across sources, since baseline difficulty and topic
coverage differ sharply between them (see the dashboard's "Accuracy by Climate Topic" section
for why)

**Topic coverage:** every item across all five sources carries `liu_topic` (Liu et al. 26-topic
taxonomy), native for ClimaQA-Gold/Climate-FEVER/generated MCQ, LLM-backfilled (`pipeline/
05_tag_topics.py`) for PIRA 2.0 and CLINB, which don't carry it natively.

**Building the English master:**
```bash
uv run python pipeline/01_select.py           # ClimaQA-Gold + Climate-FEVER + PIRA 2.0
uv run python pipeline/01b_load_clinb.py      # CLINB (requires Kaggle credentials)
uv run python pipeline/02a_chunk_pdfs.py      # chunk IPCC AR6 report PDFs
uv run python pipeline/02b_prefilter_chunks.py
uv run python pipeline/02_generate_mcq_v2.py  # generate + verify candidate MCQ
uv run python scripts/qc_sample.py            # human accept/reject pass
uv run python scripts/extract_accepted.py     # -> accepted_v2.jsonl
uv run python pipeline/05_tag_topics.py       # backfill liu_topic everywhere
uv run python pipeline/03_freeze.py           # merges + validates + writes manifest
```

## Multilingual translation pipeline

**Core 4 languages - translated, hybrid-built, and evaluated:** Dutch, Portuguese, Hindi,
Swahili.

**8 extension languages - translated, hybrid-built, QE-scored; evaluated:**
Chinese, Spanish, French, Modern Standard Arabic, Bengali, Indonesian, Russian, Urdu.

Every language uses the same pipeline:
1. **NLLB-200 3.3B** (`cluster/translate_pilot.py`, Myriad HPC, local GPU inference) --
   per-sentence-fragment translation to avoid a confirmed silent-truncation bug on long
   fragments; primary translator, run on institutional compute rather than metered gateway cost.
2. **Qwen3-235B** (`pipeline/04_translate_qwen.py`, gateway) -- whole-item JSON-structured
   translation (question + all options together in one call) for cross-option coherence;
   cross-check translator. Never the same model as any SUT, generator, or judge role.
3. **CometKiwi QE scoring** (`pipeline/05_tq_score.py`, reference-free, per-fragment) for both
   translators independently.
4. **Hybrid selection** (`pipeline/05c_build_hybrid_translation.py`) -- per item, take
   whichever translator scored higher; zero new translation calls. This is what evaluation
   actually reads (`data/full_<lang>_hybrid.jsonl`). Per-item margin matters: a large-margin
   hybrid win reliably catches real mistranslations, but roughly half of all per-item wins
   are within CometKiwi's noise band (<0.01) and shouldn't be read as a confident quality
   judgment on their own -- aggregate means are still valid either way.
5. **Reference-based validation** (`pipeline/05b_tq_score_pira_reference.py`) -- COMET scored
   against PIRA's own human Portuguese translations, the only point in the pipeline with a real
   human reference; used to confirm CometKiwi can be trusted elsewhere.

**Evidence handling (Climate-FEVER):** claims are translated, but their supporting evidence
sentences are deliberately carried through *untranslated* (English) rather than machine-translated
-- evidence is longer, more information-dense text where a translation error would silently
corrupt the ground truth being reasoned over, and authoritative climate documentation is
disproportionately English-first in practice anyway. Both translator scripts originally dropped
this field outright (a real bug, not the intended scope-limited behaviour); fixed, and backfilled
into already-translated files via `scripts/backfill_evidence.py` (no re-translation needed, since
evidence text never changes across languages).

## Models

Configured in `src/models.py` under `MODEL_REGISTRY`. To add a model: one new dict entry.
SUT panel and pipeline-utility models (generator/verifier/judge/translator) are kept in
explicitly separate registry sections -- a model evaluated as a SUT is never reused for
generation, verification, translation, or judging.

**SUT panel** (evaluated models):

| ID | Notes |
|----|-------|
| `claude-haiku` | frontier |
| `qwen3-32b`, `llama3-8b`, `ministral-3-8b`, `llama3-70b`, `mistral-large`, `gpt-oss-20b`, `deepseek-v3`, `nova-pro` | open-weight, mixed size tiers |
| `gemma3-12b` | full multilingual coverage across all 12 languages |
| `claude-opus-5`, `gpt-5.6-sol` | frontier tier, deliberately non-Qwen-family, for the multilingual frontier comparison |

**Pipeline-utility models** (never evaluated as a SUT): `qwen3-235b-generator` (MCQ generation +
Qwen translation), `claude-sonnet` (verification, topic tagging, freeform judge).

Reasoning-tier models (`claude-opus-5`, `gpt-oss-20b`) have a confirmed failure mode where
invisible internal reasoning can consume the entire `max_tokens` budget before any visible
output -- mitigated with raised budgets and a graceful empty-string fallback rather than a
crash, though not fully eliminated.

All calls go through the same gateway. `long_endpoint: True` routes to `GATEWAY_URL_LONG`.

## Running evals

```bash
# English, full panel, with MCQ position-bias debiasing
uv run python pipeline/06_run_eval.py --model gemma3-12b --debias-mcq

# a translated/hybrid language set
guv run python pipeline/06_run_eval.py --model gemma3-12b --input data/full_nl_hybrid.jsonl --debias-mcq

# pilot / smoke test
uv run python pipeline/06_run_eval.py --model gemma3-12b --n 40 --dry-run

# freeform (CLINB) LLM-judge scoring, on a completed run
uv run python pipeline/07_judge_freeform.py --run-dir results/<run_dir_name>

# validate judge reliability against a human rater
uv run python scripts/sample_judge_review.py --run-dir results/<run_dir_name>
#  ... fill in the your_* columns in the generated CSV, then:
uv run python scripts/score_judge_agreement.py --sample results/<run_dir_name>/judge_qc_sample.csv
```

`--debias-mcq` runs every MCQ item once per cyclic rotation of its own options (4x/5x call
cost) so a model's letter-position preference can't inflate its accuracy, and reports a
position-bias diagnostic table alongside the normal per-source summary.

## Results layout

```
results/
  <ts>_<model_id>_<lang>_<pilot|full>/
    config.json               model, master file, seed, item counts by source, debias flag
    invocation.txt            exact command line + resolved input path (provenance, added
                               after a run-directory-collision incident under concurrent launches)
    responses.jsonl           raw item + prompt + response (one line per cyclic rotation for
                               debiased MCQ items)
    scores.jsonl              scored records; one aggregated record per item even under
                               --debias-mcq (see scorer.aggregate_circular_mcq)
    summary.json               per-source accuracy breakdown
    mcq_position_bias.json    (--debias-mcq only) accuracy by which letter the gold answer
                               occupies, grouped by option count
    freeform_judged.jsonl     (after 07_judge_freeform.py) per-item 6-dimension rubric scores
    freeform_summary.json     mean judge score per dimension
    judge_qc_sample.csv/.md   (after sample_judge_review.py) human-validation sample

  tq_scores/
    <lang>_<nllb|qwen>_cometkiwi.jsonl          per-item reference-free QE scores
    <lang>_<nllb|qwen>_cometkiwi_summary.json   mean score, flagged-low count, by-source breakdown
    <lang>_hybrid_summary.json                  hybrid selection counts + mean CometKiwi
    pira_pt_reference.jsonl / _summary.json     reference-based COMET validation
```

Run-directory naming is keyed on the item data's own `language` field (not the input filename,
which isn't consistently named) plus a collision-detection loop -- concurrent launches on
different languages can't silently collide on the same directory.

## Caching and cost tracking

Every real model call is cached in `.cache/responses.db`. Re-running with the same items costs
nothing. Cache key: `SHA-256(model_id + prompt + params)`.

Every real call appends to `.cache/cost_log.jsonl`:
```json
{"ts": "...", "model_id": "...", "cost": 0.0001, "remaining": 19.99, "input_tokens": 124, "output_tokens": 4}
```

```bash
uv run python scripts/cost_summary.py     # total spend, remaining budget, per-model token counts
uv run pytest tests/
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
