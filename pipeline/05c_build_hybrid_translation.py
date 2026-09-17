"""
Stage 5c: Build a QE-selected "hybrid" translation per language -- for each
item, take whichever translator (NLLB-200 3.3B or Qwen3-235B) scored higher
on reference-free CometKiwi (pipeline/05_tq_score.py). This is a standard,
well-established technique (QE-based system combination / segment-level
selection -- e.g. Fernandes et al., "Quality-Aware Decoding for Neural
Machine Translation"; a primary current approach is "select the best output
among multiple LLM/MT systems using QE"). Needs zero new translation calls --
it's a pure selection over data already produced and scored.

Reads:
    data/full_<lang>_nllb.jsonl / full_nl_181423.jsonl  (NLLB)
    data/full_<lang>_qwen.jsonl                          (Qwen)
    results/tq_scores/<lang>_{nllb,qwen}_cometkiwi.jsonl (per-item scores)

Writes:
    data/full_<lang>_hybrid.jsonl
    results/tq_scores/<lang>_hybrid_summary.json

Usage:
    uv run python pipeline/05c_build_hybrid_translation.py
    uv run python pipeline/05c_build_hybrid_translation.py --lang sw
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "tq_scores"

TRANSLATOR_FILES = {
    "nl": {"nllb": ROOT / "data" / "full_nl_181423.jsonl", "qwen": ROOT / "data" / "full_nl_qwen.jsonl"},
    "pt": {"nllb": ROOT / "data" / "full_pt_nllb.jsonl", "qwen": ROOT / "data" / "full_pt_qwen.jsonl"},
    "hi": {"nllb": ROOT / "data" / "full_hi_nllb.jsonl", "qwen": ROOT / "data" / "full_hi_qwen.jsonl"},
    "sw": {"nllb": ROOT / "data" / "full_sw_nllb.jsonl", "qwen": ROOT / "data" / "full_sw_qwen.jsonl"},
    # extension languages
    "zh": {"nllb": ROOT / "data" / "full_zh_nllb.jsonl", "qwen": ROOT / "data" / "full_zh_qwen.jsonl"},
    "es": {"nllb": ROOT / "data" / "full_es_nllb.jsonl", "qwen": ROOT / "data" / "full_es_qwen.jsonl"},
    "fr": {"nllb": ROOT / "data" / "full_fr_nllb.jsonl", "qwen": ROOT / "data" / "full_fr_qwen.jsonl"},
    "ar": {"nllb": ROOT / "data" / "full_ar_nllb.jsonl", "qwen": ROOT / "data" / "full_ar_qwen.jsonl"},
    "bn": {"nllb": ROOT / "data" / "full_bn_nllb.jsonl", "qwen": ROOT / "data" / "full_bn_qwen.jsonl"},
    "id": {"nllb": ROOT / "data" / "full_id_nllb.jsonl", "qwen": ROOT / "data" / "full_id_qwen.jsonl"},
    "ru": {"nllb": ROOT / "data" / "full_ru_nllb.jsonl", "qwen": ROOT / "data" / "full_ru_qwen.jsonl"},
    "ur": {"nllb": ROOT / "data" / "full_ur_nllb.jsonl", "qwen": ROOT / "data" / "full_ur_qwen.jsonl"},
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_scores(lang: str, translator: str) -> dict[str, float]:
    """Keyed by translation_of (the ENGLISH item_id) -- matches how build_hybrid
    iterates, not the translated item's own (language-suffixed) item_id."""
    path = RESULTS_DIR / f"{lang}_{translator}_cometkiwi.jsonl"
    rows = load_jsonl(path)
    return {r["translation_of"]: r["mean_cometkiwi"] for r in rows}


def build_hybrid(lang: str) -> None:
    files = TRANSLATOR_FILES[lang]
    translated = {t: {d["translation_of"]: d for d in load_jsonl(p)} for t, p in files.items()}
    scores = {t: load_scores(lang, t) for t in files}

    all_source_ids = set(translated["nllb"]) | set(translated["qwen"])
    print(f"=== {lang} ===")
    print(f"  nllb: {len(translated['nllb'])} items, qwen: {len(translated['qwen'])} items, "
          f"union: {len(all_source_ids)}")

    hybrid = []
    winning_scores = []
    chosen_count = {"nllb": 0, "qwen": 0}
    no_score_fallback = 0

    for source_id in sorted(all_source_ids):
        candidates = {t: translated[t].get(source_id) for t in files if translated[t].get(source_id) is not None}
        if len(candidates) == 1:
            translator = next(iter(candidates))
            hybrid.append(candidates[translator])
            chosen_count[translator] += 1
            score = scores[translator].get(source_id)
            if score is not None:
                winning_scores.append(score)
            continue

        # both translators have this item -- pick by CometKiwi score
        item_scores = {t: scores[t].get(source_id) for t in candidates}
        if any(s is None for s in item_scores.values()):
            # shouldn't happen if 05_tq_score.py has been run for both, but
            # fall back to nllb (the originally-designated primary) rather
            # than crash
            no_score_fallback += 1
            winner = "nllb"
        else:
            winner = max(item_scores, key=item_scores.get)
        hybrid.append(candidates[winner])
        chosen_count[winner] += 1
        if item_scores.get(winner) is not None:
            winning_scores.append(item_scores[winner])

    hybrid.sort(key=lambda d: d["translation_of"])
    out_path = ROOT / "data" / f"full_{lang}_hybrid.jsonl"
    out_path.write_text("\n".join(json.dumps(d) for d in hybrid) + "\n")

    hybrid_mean = sum(winning_scores) / len(winning_scores)

    summary = {
        "lang": lang, "n_items": len(hybrid),
        "chosen_from_nllb": chosen_count["nllb"], "chosen_from_qwen": chosen_count["qwen"],
        "no_score_fallback_to_nllb": no_score_fallback,
        "mean_cometkiwi_hybrid": hybrid_mean,
    }
    (RESULTS_DIR / f"{lang}_hybrid_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"  chosen from nllb: {chosen_count['nllb']}, from qwen: {chosen_count['qwen']}"
          f"{f' ({no_score_fallback} had no score, defaulted to nllb)' if no_score_fallback else ''}")
    print(f"  hybrid mean CometKiwi: {hybrid_mean:.3f}")
    print(f"  written: {out_path}\n")


def main(lang: str | None) -> None:
    langs = [lang] if lang else list(TRANSLATOR_FILES)
    for l in langs:
        build_hybrid(l)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=list(TRANSLATOR_FILES), default=None)
    args = parser.parse_args()
    main(args.lang)
