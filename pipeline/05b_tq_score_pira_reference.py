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

Also cross-checks against reference-free CometKiwi, to test the project's
own "divergence flag" (|reference-based COMET - reference-free CometKiwi| >
0.1) -- a check on whether the reference-free proxy used for hybrid
selection everywhere else can be trusted where no human reference exists.

IMPORTANT (bug history, fixed 2026-09): this divergence check originally
joined against the already-computed results/tq_scores/pt_{translator}_cometkiwi.jsonl
via a cross-file lookup on item_id. Two bugs made that join silently
worthless for this entire script's history: (1) that file's own "item_id"
field is the TRANSLATED item's id (e.g. "pira_mcq_A783_pt"), not the bare
English item_id used everywhere in this script -- the lookup never matched,
so cometkiwi_reference_free was None for all 200 records, forever, and "0
divergence flagged" in every historical summary was a null-safe default,
not a real result; (2) even with the key fixed, that file's mean_cometkiwi
is averaged over ALL of an item's fragments (question + every MCQ option,
6 for PIRA's 5-option format), whereas mean_comet_reference here only
covers question + gold-answer (2 fragments) -- comparing a 6-fragment mean
against a 2-fragment mean is not a fair scope match regardless of the key
fix. Both are fixed at once below by scoring CometKiwi IN-PROCESS on the
identical {src, mt} pairs already built for the COMET pass, rather than
joining against a separately-computed file -- this guarantees scope parity
by construction and removes the cross-file join (and its key-mismatch risk)
entirely. Corrected result on this 100-item set: NLLB 15/100 flagged, Qwen
20/100 flagged, mean divergence ~0.07 for both, and the direction is
systematic (CometKiwi underestimates relative to the human reference in 34
of the 35 total flagged cases) -- not the "0 flagged" previously reported.
See DECISIONS_LOG.md for the full writeup.

Runs in the SEPARATE .venv-comet (see pipeline/05_tq_score.py's docstring
for why, and the one-time HF license/token setup -- this needs BOTH
Unbabel/wmt22-comet-da and Unbabel/wmt22-cometkiwi-da accepted).

Reads:
    data/english_master_v9.jsonl
    data/full_pt_nllb.jsonl, data/full_pt_qwen.jsonl
    paulopirozelli/pira, config "default" (human PT reference, downloaded live)

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

COMET_MODEL_ID = "Unbabel/wmt22-comet-da"
COMETKIWI_MODEL_ID = "Unbabel/wmt22-cometkiwi-da"
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

    print(f"Loading {COMET_MODEL_ID} (one-time download + license acceptance -- see module docstring)...")
    comet_model = load_from_checkpoint(download_model(COMET_MODEL_ID))
    print(f"Loading {COMETKIWI_MODEL_ID}...")
    cometkiwi_model = load_from_checkpoint(download_model(COMETKIWI_MODEL_ID))

    records = []
    for translator, path in TRANSLATOR_FILES.items():
        if not path.exists():
            print(f"  {path.name}: not found, skipping {translator}")
            continue
        translated_by_source_id = {it.translation_of: it for it in load_items(path)}

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
        comet_scores = comet_model.predict(model_input, batch_size=16, gpus=0, num_workers=1).scores

        # Reference-free CometKiwi, scored on the IDENTICAL src/mt pairs (the
        # "ref" key is simply unused) -- this guarantees scope parity with
        # the COMET pass above by construction, rather than by joining
        # against a separately-computed, differently-scoped file. See
        # module docstring for the bug history this replaces.
        kiwi_input = [{"src": r["src"], "mt": r["mt"]} for r in model_input]
        kiwi_scores = cometkiwi_model.predict(kiwi_input, batch_size=16, gpus=0, num_workers=1).scores

        by_item_comet: dict[str, dict[str, float]] = {}
        by_item_kiwi: dict[str, dict[str, float]] = {}
        for (item_id, field), c_score, k_score in zip(meta, comet_scores, kiwi_scores):
            by_item_comet.setdefault(item_id, {})[field] = c_score
            by_item_kiwi.setdefault(item_id, {})[field] = k_score

        for item_id, field_scores in by_item_comet.items():
            mean_comet = sum(field_scores.values()) / len(field_scores)
            kiwi_fields = by_item_kiwi[item_id]
            mean_kiwi = sum(kiwi_fields.values()) / len(kiwi_fields)
            # signed: positive = CometKiwi underestimates quality relative
            # to the human-reference score; negative = it overestimates
            divergence = mean_comet - mean_kiwi
            records.append({
                "item_id": item_id, "translator": translator,
                **{f"comet_{f}": s for f, s in field_scores.items()},
                "mean_comet_reference": mean_comet,
                "cometkiwi_reference_free": mean_kiwi,
                "divergence": divergence,
                "divergence_flagged": abs(divergence) > DIVERGENCE_THRESHOLD,
            })

        mean_by_translator = sum(r["mean_comet_reference"] for r in records if r["translator"] == translator) / \
            sum(1 for r in records if r["translator"] == translator)
        print(f"  {translator}: {len(by_item_comet)} items scored, mean reference-based COMET = {mean_by_translator:.3f}")

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
            "mean_cometkiwi_reference_free": sum(r["cometkiwi_reference_free"] for r in rows) / len(rows),
            "mean_abs_divergence": sum(abs(r["divergence"]) for r in rows) / len(rows),
            "n_divergence_flagged": sum(1 for r in rows if r["divergence_flagged"]),
        }
    summary = {
        "comet_model": COMET_MODEL_ID, "cometkiwi_model": COMETKIWI_MODEL_ID,
        "n_overlap_items": len(overlap),
        "divergence_threshold": DIVERGENCE_THRESHOLD,
        "divergence_scope_note": "Both COMET and CometKiwi scored on the identical question+gold-answer fragments (not all MCQ options) -- see module docstring.",
        "by_translator": by_translator,
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

        # Item-level win count -- ties are counted and reported on their own
        # term, never silently folded into either translator's win count.
        by_item: dict[str, dict[str, float]] = {}
        for r in records:
            by_item.setdefault(r["item_id"], {})[r["translator"]] = r["mean_comet_reference"]
        paired = [v for v in by_item.values() if "nllb" in v and "qwen" in v]
        nllb_wins = sum(1 for v in paired if v["nllb"] > v["qwen"])
        qwen_wins = sum(1 for v in paired if v["qwen"] > v["nllb"])
        ties = sum(1 for v in paired if v["nllb"] == v["qwen"])
        print(f"Item-level wins (by reference-based COMET): "
              f"NLLB={nllb_wins}, Qwen={qwen_wins}, ties={ties} (n={len(paired)})")


if __name__ == "__main__":
    main()
