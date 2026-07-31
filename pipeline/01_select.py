"""
Stage 1: Topic-stratified selection from each source dataset.

Samples items from ClimaQA-Gold, Climate-FEVER, and PIRA 2.0 and converts
them to the unified Item schema.  Topic diversity is maximised for ClimaQA
using the Liu et al. taxonomy.  Ocean-adjacent topics (A3, B3) are
deprioritised for ClimaQA because PIRA covers that domain.

Outputs (written to data/selections/):
    climaqa_selection.jsonl        100 MCQ  (4-option, gold = lowercase letter)
    climate_fever_selection.jsonl  150 Claim (4-label: SUPPORTS/REFUTES/NEI/DISPUTED)
    pira_selection.jsonl           100 MCQ  (5-option, gold = lowercase letter)

Usage:
    uv run python pipeline/01_select.py
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset

from item_schema import Item

SEED = 42
N_CLIMAQA        = 100
N_CLIMATE_FEVER  = 150
N_PIRA           = 100

ROOT           = Path(__file__).parent.parent
SELECTIONS_DIR = ROOT / "data" / "selections"
TAXONOMY_PATH  = ROOT / "data" / "climaqa_topic_taxonomy.jsonl"

OCEAN_TOPICS = {
    "A3. Oceans, Cryosphere & Sea-Level Change",
    "B3. Marine & Coastal Ecosystem Changes",
}

CLIMATE_FEVER_LABEL_MAP = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT_ENOUGH_INFO",
    3: "DISPUTED",
}

# Custom target counts per label -- flatter than the natural distribution
# (SUPPORTS is 43% of the dataset; we reduce it to avoid class imbalance)
CLIMATE_FEVER_TARGETS = {
    "SUPPORTS":        40,
    "NOT_ENOUGH_INFO": 35,
    "REFUTES":         40,
    "DISPUTED":        35,
}

CLIMAQA_LICENSE       = "CC-BY-4.0"
PIRA_LICENSE          = "CC-BY-4.0"
CLIMATE_FEVER_LICENSE = "unknown"   # not stated on HF; check paper before publishing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_taxonomy() -> dict[str, list[str]]:
    """Return {taxonomy_key: [topic, ...]} for all ClimaQA-Gold items."""
    mapping: dict[str, list[str]] = {}
    with TAXONOMY_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            mapping[row["id"]] = row.get("Final_Topics") or []
    return mapping


def parse_climaqa_options(options_str: str) -> dict[str, str]:
    """Parse ClimaQA's 'a) text\n---\nb) text' format into {letter: text}."""
    result: dict[str, str] = {}
    for block in options_str.split("--------------------"):
        block = block.strip()
        m = re.match(r"^([a-d])\)\s*(.+)$", block, re.DOTALL)
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result


def diversity_sample(
    candidates: list[tuple[int, list[str]]],
    n: int,
    seed: int,
) -> list[int]:
    """
    Round-robin topic-diversity sampling.

    candidates: list of (index, topics) where ocean topics are already removed.
    Returns up to n indices, biased toward underrepresented topics.
    """
    rng = random.Random(seed)

    topic_to_idxs: dict[str, list[int]] = defaultdict(list)
    untagged: list[int] = []

    for idx, topics in candidates:
        if topics:
            for t in set(topics):
                topic_to_idxs[t].append(idx)
        else:
            untagged.append(idx)

    for bucket in topic_to_idxs.values():
        rng.shuffle(bucket)
    rng.shuffle(untagged)

    # Rarest topics first
    topic_order = sorted(topic_to_idxs, key=lambda t: len(topic_to_idxs[t]))
    iters = {t: iter(topic_to_idxs[t]) for t in topic_order}

    selected: list[int] = []
    seen: set[int] = set()

    # Phase 1: round-robin across topics
    exhausted: set[str] = set()
    while len(selected) < n and len(exhausted) < len(topic_order):
        for t in topic_order:
            if t in exhausted or len(selected) >= n:
                continue
            for idx in iters[t]:
                if idx not in seen:
                    selected.append(idx)
                    seen.add(idx)
                    break
            else:
                exhausted.add(t)

    # Phase 2: fill remainder from untagged items
    for idx in untagged:
        if len(selected) >= n:
            break
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)

    return selected[:n]


def stratified_sample(
    items: list[dict],
    label_key: str,
    n: int,
    seed: int,
    custom_targets: dict[str, int] | None = None,
) -> list[dict]:
    """
    Stratified sample of n items across label_key values.
    If custom_targets is given, use those counts directly (must sum to n).
    Otherwise samples proportionally to the natural label distribution.
    """
    rng = random.Random(seed)

    if custom_targets is not None:
        targets = custom_targets
    else:
        counts = Counter(item[label_key] for item in items)
        total  = len(items)
        targets = {
            label: max(1, round(n * count / total))
            for label, count in counts.items()
        }
        while sum(targets.values()) > n:
            largest = max(targets, key=targets.__getitem__)
            targets[largest] -= 1
        while sum(targets.values()) < n:
            largest = max(targets, key=targets.__getitem__)
            targets[largest] += 1

    by_label: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_label[item[label_key]].append(item)

    result: list[dict] = []
    for label, target in targets.items():
        pool = list(by_label[label])
        rng.shuffle(pool)
        result.extend(pool[:target])
    return result


def write_selection(items: list[Item], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(it.to_dict()) for it in items) + "\n")
    print(f"  -> {path.name}  ({len(items)} items)")


# ---------------------------------------------------------------------------
# Source selectors
# ---------------------------------------------------------------------------

def select_climaqa(taxonomy: dict[str, list[str]]) -> list[Item]:
    print("Loading ClimaQA-Gold MCQ...")
    ds = load_dataset("Rose-STL-Lab/ClimaQA", "Gold", trust_remote_code=True)
    mcq = ds["mcq"]

    candidates: list[tuple[int, list[str]]] = []
    for idx in range(len(mcq)):
        key = f"ClimaQA_Gold_mcq_{idx}"
        topics = taxonomy.get(key, [])
        # Exclude items whose topics are exclusively ocean -- keep if any non-ocean topic exists
        non_ocean = [t for t in topics if t not in OCEAN_TOPICS]
        if topics and not non_ocean:
            continue   # all topics are ocean -- skip
        candidates.append((idx, non_ocean or topics))

    print(f"  Eligible after ocean filter: {len(candidates)}/{len(mcq)}")

    selected_indices = diversity_sample(candidates, N_CLIMAQA, SEED)

    items: list[Item] = []
    for idx in selected_indices:
        row = mcq[idx]
        key = f"ClimaQA_Gold_mcq_{idx}"
        topics = [t for t in taxonomy.get(key, []) if t not in OCEAN_TOPICS]
        items.append(Item(
            item_id     = f"climaqa_mcq_{idx:04d}",
            source      = "climaqa_gold",
            license     = CLIMAQA_LICENSE,
            item_type   = "mcq",
            language    = "en",
            question    = row["Question"],
            gold        = row["Answer"].strip().lower(),
            options     = parse_climaqa_options(row["Options"]),
            topic       = topics,
            complexity  = row.get("Complexity"),
            source_doc_id = key,
        ))

    # Coverage report
    covered = {t for it in items for t in it.topic}
    print(f"  Topics covered: {len(covered)}/26")
    return items


def select_climate_fever(taxonomy: dict[str, list[str]]) -> list[Item]:
    print("Loading Climate-FEVER...")
    ds = load_dataset("tdiggelm/climate_fever", trust_remote_code=True)
    split_name = list(ds.keys())[0]
    rows = list(ds[split_name])

    # Tag each row with taxonomy topics (join key = Climate_FEVER_{claim_id})
    for row in rows:
        row["_topics"]   = taxonomy.get(f"Climate_FEVER_{row['claim_id']}", [])
        row["label_str"] = CLIMATE_FEVER_LABEL_MAP[row["claim_label"]]

    label_counts = Counter(r["label_str"] for r in rows)
    print(f"  Label distribution: {dict(label_counts)}")
    tagged = sum(1 for r in rows if r["_topics"])
    print(f"  Taxonomy coverage: {tagged}/{len(rows)}")

    # Per-label topic-diversity sampling -- holds label targets fixed while
    # maximising topic spread within each bucket
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label_str"]].append(row)

    selected_rows: list[dict] = []
    for label, target in CLIMATE_FEVER_TARGETS.items():
        bucket = by_label[label]
        candidates = [(i, bucket[i]["_topics"]) for i in range(len(bucket))]
        chosen = diversity_sample(candidates, target, SEED)
        selected_rows.extend(bucket[i] for i in chosen)

    sampled_counts = Counter(r["label_str"] for r in selected_rows)
    print(f"  Sampled label distribution: {dict(sampled_counts)}")
    covered = {t for r in selected_rows for t in r["_topics"]}
    print(f"  Topics covered: {len(covered)}/26")

    items: list[Item] = []
    for row in selected_rows:
        items.append(Item(
            item_id   = f"climate_fever_{row['claim_id']}",
            source    = "climate_fever",
            license   = CLIMATE_FEVER_LICENSE,
            item_type = "claim",
            language  = "en",
            question  = row["claim"],
            gold      = row["label_str"],
            topic     = row["_topics"],
        ))
    return items


def select_pira() -> list[Item]:
    print("Loading PIRA 2.0 MCQA (test split)...")
    ds = load_dataset("paulopirozelli/pira", "mcqa", trust_remote_code=True)
    test_rows = list(ds["test"])   # 227 items

    rng = random.Random(SEED)
    rng.shuffle(test_rows)
    sampled = test_rows[:N_PIRA]

    items: list[Item] = []
    for row in sampled:
        options = {
            "a": row["A"],
            "b": row["B"],
            "c": row["C"],
            "d": row["D"],
            "e": row["E"],
        }
        items.append(Item(
            item_id      = f"pira_mcq_{row['id']}",
            source       = "pira2",
            license      = PIRA_LICENSE,
            item_type    = "mcq",
            language     = "en",
            question     = row["question"],
            gold         = row["alternative"].strip().lower(),
            options      = options,
            source_doc_id = row.get("id"),
        ))
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Liu et al. topic taxonomy...")
    taxonomy = load_taxonomy()
    print(f"  Taxonomy entries: {len(taxonomy)}")

    print()
    climaqa_items = select_climaqa(taxonomy)

    print()
    fever_items = select_climate_fever(taxonomy)

    print()
    pira_items = select_pira()

    print()
    print("Writing selections...")
    write_selection(climaqa_items, SELECTIONS_DIR / "climaqa_selection.jsonl")
    write_selection(fever_items,   SELECTIONS_DIR / "climate_fever_selection.jsonl")
    write_selection(pira_items,    SELECTIONS_DIR / "pira_selection.jsonl")

    total = len(climaqa_items) + len(fever_items) + len(pira_items)
    print(f"\nDone.  {total} items across 3 sources  (generated MCQ + CLINB pending)")


if __name__ == "__main__":
    main()
