"""
Stage 5: Reference-free translation quality scoring (CometKiwi QE) across
every translated file and every translator (NLLB-200 3.3B and Qwen3-235B
cross-check), per project decision.

Runs in a SEPARATE venv (.venv-comet, not the main project uv env) --
unbabel-comet requires transformers<5.0, which conflicts with this project's
transformers>=5.15.0 (needed for the NLLB/gateway pipeline). See
.venv-comet setup in the repo root.

One-time setup before this will run:
    1. Accept the license at https://huggingface.co/Unbabel/wmt22-cometkiwi-da
       (gated model -- requires being logged into a HF account).
    2. export HF_TOKEN=<a token from https://huggingface.co/settings/tokens>

Scores at the same granularity translation happened at (question + each MCQ
option, or the claim, or the freeform question) -- one CometKiwi call per
fragment, {"src": english_text, "mt": translated_text}, no reference needed.
Per-item score is the mean across that item's fragments.

Reads:
    data/english_master_v9.jsonl   (source text, joined via translation_of)
    Translated files (see LANG_FILES below)

Writes:
    results/tq_scores/<lang>_<translator>_cometkiwi.jsonl   per-item scores
    results/tq_scores/<lang>_<translator>_cometkiwi_summary.json

Usage (from the repo root, using the separate venv):
    .venv-comet/bin/python pipeline/05_tq_score.py
    .venv-comet/bin/python pipeline/05_tq_score.py --lang pt --translator nllb
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from item_schema import Item

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "tq_scores"
MASTER_PATH = ROOT / "data" / "english_master_v9.jsonl"

MODEL_ID = "Unbabel/wmt22-cometkiwi-da"
LOW_SCORE_THRESHOLD = 0.7   # project decision: flag items below this

# (lang, translator) -> file. Core four languages have both translators;
# the 8 extension languages (zh/es/fr/ar/bn/id/ru/ur) were added once their
# NLLB (cluster/translate_full_nllb33b_extension.job) and Qwen
# (pipeline/04_translate_qwen.py) translations landed -- see
# cluster/translate_full_nllb33b_extension.job's docstring for the language list.
LANG_FILES = {
    ("nl", "nllb"): ROOT / "data" / "full_nl_181423.jsonl",
    ("nl", "qwen"): ROOT / "data" / "full_nl_qwen.jsonl",
    ("pt", "nllb"): ROOT / "data" / "full_pt_nllb.jsonl",
    ("pt", "qwen"): ROOT / "data" / "full_pt_qwen.jsonl",
    ("hi", "nllb"): ROOT / "data" / "full_hi_nllb.jsonl",
    ("hi", "qwen"): ROOT / "data" / "full_hi_qwen.jsonl",
    ("sw", "nllb"): ROOT / "data" / "full_sw_nllb.jsonl",
    ("sw", "qwen"): ROOT / "data" / "full_sw_qwen.jsonl",
    ("zh", "nllb"): ROOT / "data" / "full_zh_nllb.jsonl",
    ("zh", "qwen"): ROOT / "data" / "full_zh_qwen.jsonl",
    ("es", "nllb"): ROOT / "data" / "full_es_nllb.jsonl",
    ("es", "qwen"): ROOT / "data" / "full_es_qwen.jsonl",
    ("fr", "nllb"): ROOT / "data" / "full_fr_nllb.jsonl",
    ("fr", "qwen"): ROOT / "data" / "full_fr_qwen.jsonl",
    ("ar", "nllb"): ROOT / "data" / "full_ar_nllb.jsonl",
    ("ar", "qwen"): ROOT / "data" / "full_ar_qwen.jsonl",
    ("bn", "nllb"): ROOT / "data" / "full_bn_nllb.jsonl",
    ("bn", "qwen"): ROOT / "data" / "full_bn_qwen.jsonl",
    ("id", "nllb"): ROOT / "data" / "full_id_nllb.jsonl",
    ("id", "qwen"): ROOT / "data" / "full_id_qwen.jsonl",
    ("ru", "nllb"): ROOT / "data" / "full_ru_nllb.jsonl",
    ("ru", "qwen"): ROOT / "data" / "full_ru_qwen.jsonl",
    ("ur", "nllb"): ROOT / "data" / "full_ur_nllb.jsonl",
    ("ur", "qwen"): ROOT / "data" / "full_ur_qwen.jsonl",
}


def load_items(path: Path) -> list[Item]:
    return [Item.from_dict(json.loads(l)) for l in path.read_text().splitlines() if l.strip()]


def build_fragments(source_by_id: dict[str, Item], translated: list[Item]) -> list[dict]:
    """One row per (item_id, field) fragment: {item_id, field, src, mt}."""
    fragments = []
    skipped = 0
    for it in translated:
        source = source_by_id.get(it.translation_of)
        if source is None:
            skipped += 1
            continue
        fragments.append({"item_id": it.item_id, "field": "question", "src": source.question, "mt": it.question})
        for letter, mt_text in (it.options or {}).items():
            src_text = (source.options or {}).get(letter)
            if src_text is None:
                continue
            fragments.append({"item_id": it.item_id, "field": f"option_{letter}", "src": src_text, "mt": mt_text})
    if skipped:
        print(f"  WARNING: {skipped} translated items had no matching source in {MASTER_PATH.name}, skipped")
    return fragments


def score_file(lang: str, translator: str, path: Path, model) -> None:
    if not path.exists():
        print(f"  {path.name}: not found, skipping")
        return

    master_items = load_items(MASTER_PATH)
    source_by_id = {it.item_id: it for it in master_items}
    translated = load_items(path)

    fragments = build_fragments(source_by_id, translated)
    print(f"  {path.name}: {len(translated)} items, {len(fragments)} fragments to score")

    model_input = [{"src": f["src"], "mt": f["mt"]} for f in fragments]
    # num_workers=1 (not the default 0) works around a comet bug on Apple
    # Silicon: it unconditionally sets multiprocessing_context="fork" whenever
    # MPS is available, which PyTorch's DataLoader rejects unless num_workers>0.
    output = model.predict(model_input, batch_size=16, gpus=0, num_workers=1)
    scores = output.scores

    by_item: dict[str, list[float]] = {}
    for f, score in zip(fragments, scores):
        by_item.setdefault(f["item_id"], []).append(score)

    per_item_records = []
    for it in translated:
        item_scores = by_item.get(it.item_id)
        if item_scores is None:
            continue
        mean_score = sum(item_scores) / len(item_scores)
        per_item_records.append({
            "item_id": it.item_id, "translation_of": it.translation_of, "source": it.source,
            "item_type": it.item_type, "mean_cometkiwi": mean_score,
            "n_fragments": len(item_scores), "flagged_low": mean_score < LOW_SCORE_THRESHOLD,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{lang}_{translator}_cometkiwi.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in per_item_records) + "\n")

    by_source: dict[str, list[float]] = {}
    for r in per_item_records:
        by_source.setdefault(r["source"], []).append(r["mean_cometkiwi"])
    summary = {
        "lang": lang, "translator": translator, "model": MODEL_ID,
        "n_items": len(per_item_records),
        "mean_cometkiwi_overall": sum(r["mean_cometkiwi"] for r in per_item_records) / len(per_item_records),
        "n_flagged_low": sum(1 for r in per_item_records if r["flagged_low"]),
        "low_score_threshold": LOW_SCORE_THRESHOLD,
        "by_source": {s: sum(v) / len(v) for s, v in by_source.items()},
    }
    (RESULTS_DIR / f"{lang}_{translator}_cometkiwi_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"    mean CometKiwi: {summary['mean_cometkiwi_overall']:.3f}  "
          f"flagged (<{LOW_SCORE_THRESHOLD}): {summary['n_flagged_low']}/{summary['n_items']}")
    print(f"    by source: { {k: round(v, 3) for k, v in summary['by_source'].items()} }")


def main(lang: str | None, translator: str | None) -> None:
    from comet import download_model, load_from_checkpoint

    print(f"Loading {MODEL_ID} (one-time download + license acceptance required -- see module docstring)...")
    model_path = download_model(MODEL_ID)
    model = load_from_checkpoint(model_path)

    targets = [
        (l, t, p) for (l, t), p in LANG_FILES.items()
        if (lang is None or l == lang) and (translator is None or t == translator)
    ]
    if not targets:
        print(f"No matching (lang, translator) combination for lang={lang!r} translator={translator!r}")
        sys.exit(1)

    for l, t, p in targets:
        print(f"\n=== {l} / {t} ===")
        score_file(l, t, p, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["nl", "pt", "hi", "sw", "zh", "es", "fr", "ar", "bn", "id", "ru", "ur"],
                        default=None, help="Only score this language (default: all)")
    parser.add_argument("--translator", choices=["nllb", "qwen"], default=None,
                        help="Only score this translator (default: all)")
    args = parser.parse_args()
    main(args.lang, args.translator)
