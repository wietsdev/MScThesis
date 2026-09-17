"""
CometKiwi QE scoring across all 5 rungs of a translation quality ladder (see
scripts/build_quality_ladder.py). Same fragment-level scoring as
pipeline/05_tq_score.py / scripts/score_sonnet_sample.py, generalized to N
named rungs on an identical item set.

Must run in the separate .venv-comet environment.

Reads:
    data/english_master_v9.jsonl, data/ladder_{lang}_{1..5}_{tag}.jsonl

Writes:
    results/tq_scores/ladder_{lang}_comparison.json
    (per-rung mean CometKiwi + per-item scores)

Usage:
    .venv-comet/bin/python scripts/score_ladder.py --lang sw
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from item_schema import Item

ROOT = Path(__file__).parent.parent
MASTER_PATH = ROOT / "data" / "english_master_v9.jsonl"
MODEL_ID = "Unbabel/wmt22-cometkiwi-da"

RUNGS = [
    (1, "sonnet"), (2, "hybrid"), (3, "loser"), (4, "degraded"),
    (5, "negation"), (6, "number"), (7, "technical"), (8, "severe"),
]


def load_items(path: Path) -> list[Item]:
    return [Item.from_dict(json.loads(l)) for l in path.read_text().splitlines() if l.strip()]


def build_fragments(source_by_id: dict[str, Item], translated: list[Item]) -> list[dict]:
    fragments = []
    for it in translated:
        source = source_by_id.get(it.translation_of)
        if source is None:
            continue
        fragments.append({"item_id": it.item_id, "field": "question", "src": source.question, "mt": it.question})
        for letter, mt_text in (it.options or {}).items():
            src_text = (source.options or {}).get(letter)
            if src_text is None:
                continue
            fragments.append({"item_id": it.item_id, "field": f"option_{letter}", "src": src_text, "mt": mt_text})
    return fragments


def score_fragments(fragments: list[dict], model) -> dict[str, float]:
    if not fragments:
        return {}
    model_input = [{"src": f["src"], "mt": f["mt"]} for f in fragments]
    output = model.predict(model_input, batch_size=16, gpus=0, num_workers=1)
    by_item: dict[str, list[float]] = {}
    for f, score in zip(fragments, output.scores):
        by_item.setdefault(f["item_id"], []).append(score)
    return {item_id: sum(v) / len(v) for item_id, v in by_item.items()}


def main(lang: str, rungs: list[int] | None) -> None:
    from comet import download_model, load_from_checkpoint

    print(f"Loading {MODEL_ID}...")
    model = load_from_checkpoint(download_model(MODEL_ID))

    master_items = load_items(MASTER_PATH)
    source_by_id = {it.item_id: it for it in master_items}

    out_path = ROOT / "results" / "tq_scores" / f"ladder_{lang}_comparison.json"
    rung_results = json.loads(out_path.read_text()) if out_path.exists() else {}

    targets = [(n, t) for n, t in RUNGS if not rungs or n in rungs]
    for num, tag in targets:
        path = ROOT / "data" / f"ladder_{lang}_{num}_{tag}.jsonl"
        items = load_items(path)
        fragments = build_fragments(source_by_id, items)
        scores = score_fragments(fragments, model)

        valid_scores = [scores[it.item_id] for it in items if it.item_id in scores]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        print(f"  {num}_{tag}: n={len(valid_scores)}  mean CometKiwi={mean_score:.3f}" if mean_score is not None
              else f"  {num}_{tag}: n=0  mean CometKiwi=n/a")

        rung_results[f"{num}_{tag}"] = {
            "n_items": len(valid_scores), "mean_cometkiwi": mean_score,
            "per_item": {it.translation_of: scores.get(it.item_id) for it in items},
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rung_results, indent=2))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    parser.add_argument("--rungs", type=int, nargs="+", default=None,
                        help="Only (re)score these rung numbers (default: all, merges into existing output)")
    args = parser.parse_args()
    main(args.lang, args.rungs)
