"""
Stage 1b: Load all 200 CLINB freeform questions from Kaggle.

CLINB has no single gold answer (multiple hybrid candidates, rubric-based eval).
Items enter the benchmark unscored; gold is set to "" until the LLM judge is built.

Fields mapped from CLINB:
    question_id -> item_id (clinb_<id>)
    question    -> question
    level       -> ipcc_confidence  (High Confidence | Advanced | Open)
    wg          -> source_doc_id    (IPCC Working Group(s): I | II | III)
    topic       -> topic            (CLINB-specific: Detection/Extremes/Finance/
                                     Impacts/Pathways/Scenarios -- not Liu taxonomy)

Output:
    data/selections/clinb_selection.jsonl   (200 freeform items)

Usage:
    uv run python pipeline/01b_load_clinb.py

Requires Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_* env vars).
"""

import ast
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import kagglehub
from kagglehub import KaggleDatasetAdapter

from item_schema import Item

ROOT           = Path(__file__).parent.parent
SELECTIONS_DIR = ROOT / "data" / "selections"
OUTPUT_PATH    = SELECTIONS_DIR / "clinb_selection.jsonl"

CLINB_LICENSE = "CC-BY-4.0"


def load_clinb_df():
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "deepmind/clinb-questions",
        "CLINB-questions.csv",
    )


def df_to_items(df) -> list[Item]:
    items: list[Item] = []
    for _, row in df.iterrows():
        topics   = ast.literal_eval(row["topic"])    # list of strings
        wg_list  = ast.literal_eval(row["wg"])       # e.g. ["I", "II"]
        level    = ast.literal_eval(row["level"])    # e.g. ["High Confidence"]

        items.append(Item(
            item_id          = f"clinb_{row['question_id']}",
            source           = "clinb",
            license          = CLINB_LICENSE,
            item_type        = "freeform",
            language         = "en",
            question         = row["question"],
            gold             = "",           # no single gold; scored by LLM judge
            topic            = topics,       # CLINB-specific, not Liu taxonomy
            ipcc_confidence  = level[0] if level else None,
            source_doc_id    = "IPCC-WG-" + "+".join(wg_list),
        ))
    return items


def main() -> None:
    print("Downloading CLINB from Kaggle...")
    df = load_clinb_df()
    print(f"  Loaded {len(df)} rows")

    items = df_to_items(df)

    # Distribution report
    from collections import Counter
    level_counts = Counter(it.ipcc_confidence for it in items)
    topic_counts = Counter(t for it in items for t in it.topic)
    wg_counts    = Counter(it.source_doc_id for it in items)
    print(f"  Level: {dict(level_counts)}")
    print(f"  Topic: {dict(topic_counts)}")
    print(f"  WG:    {dict(wg_counts)}")

    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(it.to_dict()) for it in items) + "\n"
    )
    print(f"\nWritten: {OUTPUT_PATH}  ({len(items)} items)")
    print("Run pipeline/03_freeze.py to update the master (will auto-increment version).")


if __name__ == "__main__":
    main()
