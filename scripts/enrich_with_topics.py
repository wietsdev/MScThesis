"""
Enrich scores.jsonl files with topic labels from the ClimaQA topic taxonomy.

For each scored record, looks up 'ClimaQA_Gold_<split>_<id>' in the taxonomy
and adds a 'topics' field (list of strings, or null if not covered).
Writes scores_enriched.jsonl alongside the original scores.jsonl.

Usage:
    uv run python scripts/enrich_with_topics.py                         # most recent matrix
    uv run python scripts/enrich_with_topics.py results/matrix_<ts>    # specific matrix
    uv run python scripts/enrich_with_topics.py results/matrix_<ts>/claude-haiku_mcq/scores.jsonl
"""

import json
import sys
from pathlib import Path

TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "climaqa_topic_taxonomy.jsonl"
RESULTS_DIR   = Path(__file__).parent.parent / "results"


def load_taxonomy() -> dict:
    """Build lookup: ClimaQA_Gold_<split>_<id> -> list of topic strings."""
    taxonomy = {}
    for line in TAXONOMY_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"].startswith("ClimaQA_Gold"):
            taxonomy[r["id"]] = r["Final_Topics"]
    return taxonomy


def enrich_file(scores_path: Path, taxonomy: dict) -> None:
    config_path = scores_path.parent / "config.json"
    if not config_path.exists():
        print(f"  Skipping {scores_path} -- no config.json found")
        return

    split = json.loads(config_path.read_text())["split"]
    lines = [l for l in scores_path.read_text().splitlines() if l.strip()]
    records = [json.loads(l) for l in lines]

    enriched = []
    n_matched, n_missing = 0, 0
    for rec in records:
        key = f"ClimaQA_Gold_{split}_{rec['id']}"
        topics = taxonomy.get(key)
        if topics:
            n_matched += 1
        else:
            n_missing += 1
        enriched.append({**rec, "topics": topics})

    out_path = scores_path.parent / "scores_enriched.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in enriched) + "\n")
    print(f"  {scores_path.parent.name}: {n_matched} tagged, {n_missing} null  -> {out_path.name}")


def find_scores_files(target: Path) -> list[Path]:
    """Return all scores.jsonl files under target (matrix dir or single file)."""
    if target.is_file() and target.name == "scores.jsonl":
        return [target]
    return sorted(target.glob("*/scores.jsonl"))


def find_latest_matrix() -> Path:
    candidates = sorted(RESULTS_DIR.glob("matrix_*"))
    if not candidates:
        raise FileNotFoundError(f"No matrix runs found in {RESULTS_DIR}")
    return candidates[-1]


def main() -> None:
    if not TAXONOMY_PATH.exists():
        print(f"Taxonomy file not found: {TAXONOMY_PATH}")
        print("Copy it to data/climaqa_topic_taxonomy.jsonl first.")
        sys.exit(1)

    if len(sys.argv) == 2:
        target = Path(sys.argv[1])
    else:
        target = find_latest_matrix()
        print(f"(No path given -- using latest: {target.name})\n")

    print(f"Loading taxonomy ({TAXONOMY_PATH.name})...")
    taxonomy = load_taxonomy()
    gold_entries = sum(1 for k in taxonomy if k.startswith("ClimaQA_Gold"))
    print(f"  {gold_entries} ClimaQA Gold entries loaded\n")

    scores_files = find_scores_files(target)
    if not scores_files:
        print(f"No scores.jsonl files found under {target}")
        sys.exit(1)

    print(f"Enriching {len(scores_files)} file(s):")
    for sf in scores_files:
        enrich_file(sf, taxonomy)

    print("\nDone. Use scores_enriched.jsonl for topic-level analysis.")


if __name__ == "__main__":
    main()
