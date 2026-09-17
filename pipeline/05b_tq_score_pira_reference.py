"""
Stage 5b: Reference-based COMET scoring for the Portuguese PIRA subset,
against PIRA's own human translations -- the one place in this benchmark
with a genuine human reference (per project decision: PIRA is included
specifically as the EN/PT translation anchor).

Join: our PIRA items' source_doc_id (e.g. "A1842", from the "mcqa" HF config)
matches id_qa in the "default" config of the SAME dataset repo, which has
question_pt_origin / answer_pt_origin (human-authored Portuguese). Verified
directly: 100/100 of our PIRA items have a matching id_qa, and the English
text lines up near-verbatim (>=0.99 similarity) between the two configs, so
the join is trustworthy.

Scores BOTH translators (NLLB, Qwen) against the SAME human reference, so
you can see directly which one tracks the human translation more closely --
that's the whole point of having two translators, not just NLLB-vs-nothing.

Also cross-checks against the reference-free CometKiwi scores from
pipeline/05_tq_score.py, if those have already been run for pt -- the
project's own "divergence flag" (|reference-based COMET - reference-free
CometKiwi| > 0.1) is a check on whether the reference-free proxy can be
trusted where no reference exists (i.e. every other language/source here).

Runs in the SEPARATE .venv-comet (see pipeline/05_tq_score.py's docstring
for why, and the one-time HF license/token setup -- this also needs
Unbabel/wmt22-comet-da accepted, not just wmt22-cometkiwi-da).

Reads:
    data/english_master_v9.jsonl
    data/full_pt_nllb.jsonl, data/full_pt_qwen.jsonl
    paulopirozelli/pira, config "default" (human PT reference, downloaded live)
    results/tq_scores/pt_{nllb,qwen}_cometkiwi.jsonl   (optional, for divergence check)

Writes:
    results/tq_scores/pira_pt_reference.jsonl     per-item, per-translator, per-field scores
    results/tq_scores/pira_pt_reference_summary.json

Usage:
    .venv-comet/bin/python pipeline/05b_tq_score_pira_reference.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from item_schema import Item

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "tq_scores"
MASTER_PATH = ROOT / "data" / "english_master_v9.jsonl"

MODEL_ID = "Unbabel/wmt22-comet-da"
TRANSLATOR_FILES = {
    "nllb": ROOT / "data" / "full_pt_nllb.jsonl",
    "qwen": ROOT / "data" / "full_pt_qwen.jsonl",
}
DIVERGENCE_THRESHOLD = 0.1


def load_items(path: Path) -> list[Item]:
    return [Item.from_dict(json.loads(l)) for l in path.read_text().splitlines() if l.strip()]


def load_pira_human_reference() -> dict[str, dict]:
    from datasets import load_dataset
    ds = load_dataset("paulopirozelli/pira", "default")
    by_id_qa = {}
    for split in ds.keys():
        for row in ds[split]:
            by_id_qa[row["id_qa"]] = row
    return by_id_qa


def load_cometkiwi_scores(translator: str) -> dict[str, float]:
    path = RESULTS_DIR / f"pt_{translator}_cometkiwi.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return {r["item_id"]: r["mean_cometkiwi"] for r in rows}


def main() -> None:
    from comet import download_model, load_from_checkpoint

    master_items = {it.item_id: it for it in load_items(MASTER_PATH)}
    pira_items = [it for it in master_items.values() if it.source == "pira2"]
    print(f"PIRA items in master: {len(pira_items)}")

    human_ref = load_pira_human_reference()
    overlap = [it for it in pira_items if it.source_doc_id in human_ref]
    print(f"Overlap with human PT reference (via id_qa join): {len(overlap)}/{len(pira_items)}")
    if not overlap:
        print("No overlap found -- nothing to score.")
        return

    print(f"Loading {MODEL_ID} (one-time download + license acceptance -- see module docstring)...")
    model_path = download_model(MODEL_ID)
    model = load_from_checkpoint(model_path)

    records = []
    for translator, path in TRANSLATOR_FILES.items():
        if not path.exists():
            print(f"  {path.name}: not found, skipping {translator}")
            continue
        translated_by_source_id = {it.translation_of: it for it in load_items(path)}
        cometkiwi = load_cometkiwi_scores(translator)

        model_input, meta = [], []
        for it in overlap:
            translated = translated_by_source_id.get(it.item_id)
            if translated is None:
                continue
            ref_row = human_ref[it.source_doc_id]

            model_input.append({"src": it.question, "mt": translated.question, "ref": ref_row["question_pt_origin"]})
            meta.append((it.item_id, "question"))

            gold_letter = it.gold.lower()
            gold_text_mt = (translated.options or {}).get(gold_letter)
            if gold_text_mt and ref_row.get("answer_pt_origin"):
                model_input.append({"src": it.options[gold_letter], "mt": gold_text_mt, "ref": ref_row["answer_pt_origin"]})
                meta.append((it.item_id, "gold_answer"))

        if not model_input:
            print(f"  {translator}: no items to score")
            continue

        # num_workers=1 works around a comet/Apple-Silicon MPS + DataLoader
        # fork-context bug -- see pipeline/05_tq_score.py for the full explanation
        output = model.predict(model_input, batch_size=16, gpus=0, num_workers=1)
        scores = output.scores

        by_item: dict[str, dict[str, float]] = {}
        for (item_id, field), score in zip(meta, scores):
            by_item.setdefault(item_id, {})[field] = score

        for item_id, field_scores in by_item.items():
            mean_score = sum(field_scores.values()) / len(field_scores)
            kiwi = cometkiwi.get(item_id)
            divergence = abs(mean_score - kiwi) if kiwi is not None else None
            records.append({
                "item_id": item_id, "translator": translator,
                **{f"comet_{f}": s for f, s in field_scores.items()},
                "mean_comet_reference": mean_score,
                "cometkiwi_reference_free": kiwi,
                "divergence": divergence,
                "divergence_flagged": divergence is not None and divergence > DIVERGENCE_THRESHOLD,
            })

        mean_by_translator = sum(r["mean_comet_reference"] for r in records if r["translator"] == translator) / \
            sum(1 for r in records if r["translator"] == translator)
        print(f"  {translator}: {len(by_item)} items scored, mean reference-based COMET = {mean_by_translator:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pira_pt_reference.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    by_translator = {}
    for translator in TRANSLATOR_FILES:
        rows = [r for r in records if r["translator"] == translator]
        if not rows:
            continue
        by_translator[translator] = {
            "n_items": len(rows),
            "mean_comet_reference": sum(r["mean_comet_reference"] for r in rows) / len(rows),
            "n_divergence_flagged": sum(1 for r in rows if r["divergence_flagged"]),
        }
    summary = {
        "model": MODEL_ID, "n_overlap_items": len(overlap),
        "divergence_threshold": DIVERGENCE_THRESHOLD, "by_translator": by_translator,
    }
    (RESULTS_DIR / "pira_pt_reference_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWritten: {out_path}")
    print(f"Summary: {json.dumps(summary, indent=2)}")
    if all(t in by_translator for t in TRANSLATOR_FILES):
        nllb_mean = by_translator["nllb"]["mean_comet_reference"]
        qwen_mean = by_translator["qwen"]["mean_comet_reference"]
        better = "NLLB" if nllb_mean > qwen_mean else "Qwen"
        print(f"\n{better} tracks the human PT reference more closely on this subset "
              f"(NLLB={nllb_mean:.3f} vs Qwen={qwen_mean:.3f}).")


if __name__ == "__main__":
    main()
