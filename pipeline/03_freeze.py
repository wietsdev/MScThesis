"""
Stage 3: Freeze the English master dataset.

Merges all selection files into a single versioned JSONL file and writes a
manifest with item counts, source breakdown, and a SHA-256 hash of the output.
Once frozen, the English master should not change -- translation and eval
pipelines depend on its stability.

Expected inputs (from pipeline/01_select.py and pipeline/02_generate_mcq*.py):
    data/selections/climaqa_selection.jsonl
    data/selections/climate_fever_selection.jsonl
    data/selections/pira_selection.jsonl
    data/selections/accepted_v1.jsonl          (optional -- manually QC'd v1 generated MCQ)
    data/selections/accepted_v2.jsonl          (optional -- manually QC'd v2 generated MCQ)
    data/selections/clinb_selection.jsonl      (optional -- skip if not ready)

accepted_v1.jsonl / accepted_v2.jsonl are produced by scripts/extract_accepted.py
from the manual QC pass (scripts/qc_sample.py) over generated_mcq.jsonl /
generated_mcq_v2.jsonl. The freeze step never reads the raw generated_mcq*.jsonl
candidate pools directly -- only items a human has explicitly accepted.

generated_mcq's options are also reshuffled here (see shuffle_generated_mcq_options)
to remove a generation-time position bias (the generator model heavily favours
writing the correct answer in slot "b"). ClimaQA/PIRA are external curated
benchmarks and are left in their original option order.

Outputs:
    data/english_master_v{N}.jsonl
    data_manifest/english_master_v{N}_manifest.json

Usage:
    uv run python pipeline/03_freeze.py           # auto-increments version
    uv run python pipeline/03_freeze.py --version 1
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from item_schema import Item

ROOT           = Path(__file__).parent.parent
SELECTIONS_DIR = ROOT / "data" / "selections"
MANIFEST_DIR   = ROOT / "data_manifest"

# generated_mcq's own generation model has a strong habit of writing the
# correct option in slot "b" -- a spot-check of accepted_v1+v2 found gold
# distributed a=27.5%/b=50.0%/c=18.6%/d=3.9% (vs. ~uniform for ClimaQA/PIRA,
# which come from external curated benchmarks and are left untouched). A SUT
# model with even a mild position preference would score inflated accuracy on
# generated_mcq specifically -- which also happens to be the subset covering
# the topic gaps (D1/D3 etc.) the thesis cares most about. Fixed per-item seed
# (not a single global Random -- keeps the shuffle stable regardless of item
# order/file-split changes between freezes) so this is deterministic and
# reproducible across freezes.
SHUFFLE_SEED = 42


def shuffle_generated_mcq_options(items: list[Item]) -> None:
    for it in items:
        if it.source != "generated_mcq" or it.item_type != "mcq" or not it.options:
            continue
        letters = sorted(it.options.keys())
        texts = [it.options[l] for l in letters]
        k = len(letters)
        gold_idx = letters.index(it.gold)

        order = list(range(k))
        random.Random(f"{SHUFFLE_SEED}:{it.item_id}").shuffle(order)
        it.options = {letters[i]: texts[order[i]] for i in range(k)}
        it.gold = letters[order.index(gold_idx)]

REQUIRED_SELECTIONS = [
    "climaqa_selection.jsonl",
    "climate_fever_selection.jsonl",
    "pira_selection.jsonl",
]

OPTIONAL_SELECTIONS = [
    "accepted_v1.jsonl",
    "accepted_v2.jsonl",
    "clinb_selection.jsonl",
]

REQUIRED_FIELDS = {"item_id", "source", "license", "item_type", "language", "question", "gold"}


def load_selection(path: Path) -> list[Item]:
    items = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            items.append(Item.from_dict(json.loads(line)))
    return items


def next_version() -> int:
    existing = sorted(ROOT.glob("data/english_master_v*.jsonl"))
    if not existing:
        return 1
    last = existing[-1].stem  # e.g. "english_master_v2"
    return int(last.split("v")[-1]) + 1


def validate(items: list[Item]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for it in items:
        d = it.to_dict()
        for f in REQUIRED_FIELDS:
            # freeform items have no gold answer (scored by LLM judge later)
            if f == "gold" and it.item_type == "freeform":
                continue
            if not d.get(f):
                errors.append(f"{it.item_id}: missing required field '{f}'")
        if it.item_id in seen_ids:
            errors.append(f"Duplicate item_id: {it.item_id}")
        seen_ids.add(it.item_id)
        if it.item_type == "mcq" and it.options is None:
            errors.append(f"{it.item_id}: MCQ item missing options")
    return errors


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(version: int | None = None) -> None:
    if version is None:
        version = next_version()

    output_path   = ROOT / "data" / f"english_master_v{version}.jsonl"
    manifest_path = MANIFEST_DIR / f"english_master_v{version}_manifest.json"

    if output_path.exists():
        print(f"Error: {output_path.name} already exists. Use --version to specify a different version.")
        sys.exit(1)

    # --- Load selections ---
    all_items: list[Item] = []
    source_counts: dict[str, int] = {}

    for filename in REQUIRED_SELECTIONS:
        path = SELECTIONS_DIR / filename
        if not path.exists():
            print(f"Error: required selection file not found: {path}")
            print("Run pipeline/01_select.py first.")
            sys.exit(1)
        items = load_selection(path)
        all_items.extend(items)
        for it in items:
            source_counts[it.source] = source_counts.get(it.source, 0) + 1
        print(f"  {filename}: {len(items)} items")

    for filename in OPTIONAL_SELECTIONS:
        path = SELECTIONS_DIR / filename
        if not path.exists():
            print(f"  {filename}: not found, skipping")
            continue
        items = load_selection(path)
        all_items.extend(items)
        for it in items:
            source_counts[it.source] = source_counts.get(it.source, 0) + 1
        print(f"  {filename}: {len(items)} items")

    print(f"\nTotal items: {len(all_items)}")

    # --- Shuffle generated MCQ option order (fixes generation-time position bias) ---
    from collections import Counter
    before = Counter(it.gold for it in all_items if it.source == "generated_mcq")
    shuffle_generated_mcq_options(all_items)
    after = Counter(it.gold for it in all_items if it.source == "generated_mcq")
    print(f"\ngenerated_mcq gold letter distribution -- before shuffle: {dict(sorted(before.items()))}")
    print(f"generated_mcq gold letter distribution -- after shuffle:  {dict(sorted(after.items()))}")

    # --- Validate ---
    errors = validate(all_items)
    if errors:
        print("\nValidation errors:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    # --- Write master ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(it.to_dict()) for it in all_items) + "\n"
    )
    file_hash = sha256_of_file(output_path)

    # --- Write manifest ---
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    type_counts = {}
    for it in all_items:
        type_counts[it.item_type] = type_counts.get(it.item_type, 0) + 1

    manifest = {
        "version":      version,
        "frozen_at":    datetime.now(timezone.utc).isoformat(),
        "total_items":  len(all_items),
        "by_source":    source_counts,
        "by_type":      type_counts,
        "output_file":  output_path.name,
        "sha256":       file_hash,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nFrozen: {output_path.name}")
    print(f"  SHA-256: {file_hash}")
    print(f"  Manifest: {manifest_path.name}")
    print(f"  By source: {source_counts}")
    print(f"  By type:   {type_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args()
    main(args.version)
