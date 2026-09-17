"""
Unified item schema for the multilingual climate benchmark.

Every item in the benchmark -- regardless of source, type, or language --
is one instance of Item.  The dataclass serialises cleanly to/from JSON so
it can be used as the JSONL record type throughout the pipeline.

Required fields (must be set before an item enters the frozen English master):
    item_id, source, license, item_type, language, question, gold

Translation fields are added by pipeline/04_translate.py and default to None
for English-only items.

RAG fields (ipcc_confidence, citation) are reserved for the IPCC-RAG phase and
default to None. `evidence` is also populated for Climate-FEVER (its own
Wikipedia evidence sentences, needed for evidence-based claim verification --
see pipeline/01_select.py) ahead of that phase; same field, two provenances.
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class Item:
    # --- Identity --------------------------------------------------------
    item_id:  str   # globally unique, stable across all pipeline stages
    source:   str   # climaqa_gold | pira2 | generated_mcq | climate_fever | clinb | cards
    license:  str   # SPDX identifier, e.g. "CC-BY-4.0"

    # --- Content ---------------------------------------------------------
    item_type: str             # mcq | claim | freeform
    language:  str             # BCP-47, e.g. "en", "nl", "pt"
    question:  str             # question stem (MCQ/freeform) or claim (claim)
    gold:      str             # correct letter (MCQ), label (claim), or ref answer (freeform)
    options:   dict[str, str] | None = None  # MCQ only: {"a": "...", ...}

    # --- Classification --------------------------------------------------
    # topic holds each source's NATIVE topic system: Liu et al. taxonomy for
    # ClimaQA/Climate-FEVER/generated_mcq, but CLINB's own six-category system
    # (Detection/Extremes/Finance/Impacts/Pathways/Scenarios) for CLINB, and
    # empty for PIRA (untagged). liu_topic is the unified field -- same Liu
    # taxonomy for every source, backfilled by pipeline/05_tag_topics.py --
    # use it (not topic) for any cross-source topic-diversity analysis.
    topic:         list[str] = field(default_factory=list)
    liu_topic:     list[str] = field(default_factory=list)
    intent:        str | None = None   # from Liu taxonomy
    complexity:    str | None = None   # ClimaQA: BASE | REASONING | HYPOTHETICAL
    cards_category: int | None = None  # CARDS taxonomy 1-5 (claim items only)

    # --- Source provenance -----------------------------------------------
    source_doc_id:  str | None = None
    source_excerpt: str | None = None

    # --- RAG phase (reserved, default None) ------------------------------
    ipcc_confidence: str | None = None           # high | medium | low | very low
    evidence:        list[dict[str, Any]] | None = None  # also used by Climate-FEVER: [{"article","text"}, ...]
    citation:        str | None = None

    # --- Translation (added by pipeline/04_translate.py) -----------------
    translation_source:  str | None = None   # nllb-200 | qwen3-235b | human
    translation_of:      str | None = None   # item_id of English source
    tq_comet:            float | None = None
    tq_cometkiwi:        float | None = None
    tq_backtranslation:  str | None = None
    tq_divergence_flag:  bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Item":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
