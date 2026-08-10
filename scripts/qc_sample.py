"""
Interactive terminal QC review of generated MCQ items.

Items flagged by the automated verifier as low-confidence are shown first.
Each item shows the question, options (gold marked), supporting excerpt,
and the verifier's verdict.

Commands per item:
  a  accept
  r  reject (+ optional one-line note)
  s  skip
  q  quit and save

Results written to: data/sources/generation/qc_results.jsonl
  Each line: {item_id, decision, note, topic, confidence}

After the session, prints per-topic acceptance rates and lists items
that were rejected, for targeted follow-up.

Usage:
    uv run python scripts/qc_sample.py          # 30-item sample (low-conf first)
    uv run python scripts/qc_sample.py --n 50
    uv run python scripts/qc_sample.py --all    # review every generated item
    uv run python scripts/qc_sample.py --resume # continue from where you left off
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT         = Path(__file__).parent.parent
ITEMS_PATH   = ROOT / "data" / "selections" / "generated_mcq.jsonl"
LOG_PATH     = ROOT / "data" / "sources" / "generation" / "generation_log.jsonl"
CHUNKS_PATH  = ROOT / "data" / "sources" / "generation" / "chunks.jsonl"
RESULTS_PATH = ROOT / "data" / "sources" / "generation" / "qc_results.jsonl"

SEED = 42


def load_items() -> list[dict]:
    if not ITEMS_PATH.exists():
        print(f"No generated items found at {ITEMS_PATH}")
        print("Run pipeline/02_generate_mcq.py first.")
        sys.exit(1)
    return [json.loads(l) for l in ITEMS_PATH.read_text().splitlines() if l.strip()]


def load_chunks() -> dict[str, str]:
    """Return {chunk_id: chunk_text} for looking up full source context."""
    if not CHUNKS_PATH.exists():
        return {}
    return {
        json.loads(l)["chunk_id"]: json.loads(l)["text"]
        for l in CHUNKS_PATH.read_text().splitlines()
        if l.strip()
    }


def load_log() -> dict[str, dict]:
    """Return {item_id: log_entry} from generation_log.jsonl."""
    if not LOG_PATH.exists():
        return {}
    entries = [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l.strip()]
    return {e["item_id"]: e for e in entries}


def load_existing_results() -> set[str]:
    """Return set of item_ids already reviewed."""
    if not RESULTS_PATH.exists():
        return set()
    return {
        json.loads(l)["item_id"]
        for l in RESULTS_PATH.read_text().splitlines()
        if l.strip()
    }


def sample_items(
    items: list[dict],
    log: dict[str, dict],
    already_reviewed: set[str],
    n: int,
    review_all: bool,
) -> list[dict]:
    """
    Return items to review. Low-confidence verifier passes appear first.
    Skips items already reviewed (for --resume).
    """
    remaining = [it for it in items if it["item_id"] not in already_reviewed]

    low_conf  = [it for it in remaining if log.get(it["item_id"], {}).get("confidence") == "low"]
    other     = [it for it in remaining if it not in low_conf]

    rng = random.Random(SEED)
    rng.shuffle(other)

    ordered = low_conf + other

    if review_all:
        return ordered
    return ordered[:n]


def display_item(
    item: dict,
    log_entry: dict | None,
    idx: int,
    total: int,
    chunk_text: str | None = None,
) -> None:
    print("\n" + "=" * 64)
    print(f"  Item {idx}/{total}  —  {item['item_id']}")
    topics = ", ".join(item.get("topic", []))
    print(f"  Topic:  {topics}")
    src = item.get("source_doc_id", "")
    print(f"  Source: {src}")
    if log_entry:
        conf   = log_entry.get("confidence", "?")
        notes  = log_entry.get("notes", "")
        entail = "✓" if log_entry.get("entailed") else "✗"
        dist   = "✓" if log_entry.get("distractors_clean") else "✗"
        failed = log_entry.get("failed_checks", [])
        failed_str = f"  failed={failed}" if failed else ""
        print(f"  Verifier: entailed={entail}  distractors={dist}  confidence={conf}{failed_str}")
        if notes:
            print(f"  Notes:    {notes}")
    print("=" * 64)

    print(f"\nQUESTION:\n  {item['question']}\n")

    options = item.get("options") or {}
    gold    = item.get("gold", "").lower()
    for letter in ("a", "b", "c", "d"):
        text   = options.get(letter, "—")
        marker = "  ← GOLD" if letter == gold else ""
        print(f"  {letter.upper()}) {text}{marker}")

    excerpt = item.get("source_excerpt") or ""
    if excerpt:
        print(f'\nEXCERPT:\n  "{excerpt}"')

    if chunk_text:
        print(f"\nSOURCE CHUNK (full):\n  {chunk_text[:800]}{'...' if len(chunk_text) > 800 else ''}")


def get_decision(has_chunk: bool) -> tuple[str, str]:
    """Prompt user. Returns (decision, note) where decision in {a, r, s, c, q}."""
    chunk_hint = "  [C]hunk  " if has_chunk else ""
    prompt = f"\n[A]ccept  [R]eject  [S]kip{chunk_hint} [Q]uit > "
    while True:
        try:
            raw = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted — saving progress.")
            return "q", ""

        if raw in ("a", "r", "s", "c", "q"):
            note = ""
            if raw == "r":
                try:
                    note = input("  Rejection note (optional): ").strip()
                except (EOFError, KeyboardInterrupt):
                    pass
            return raw, note
        print("  Enter a, r, s, c, or q.")


def main(n: int, review_all: bool, resume: bool) -> None:
    items   = load_items()
    log     = load_log()
    chunks  = load_chunks()
    already = load_existing_results() if resume else set()

    queue = sample_items(items, log, already, n, review_all)

    low_conf_count = sum(
        1 for it in queue
        if log.get(it["item_id"], {}).get("confidence") == "low"
    )
    print(f"Generated items: {len(items)}")
    print(f"Queued for review: {len(queue)}  ({low_conf_count} low-confidence, shown first)")
    if already:
        print(f"Already reviewed (skipping): {len(already)}")
    if not queue:
        print("Nothing left to review.")
        return

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    skipped = 0

    for i, item in enumerate(queue, 1):
        log_entry  = log.get(item["item_id"])
        chunk_id   = item.get("source_doc_id", "")
        chunk_text = chunks.get(chunk_id)
        show_chunk = False

        while True:
            display_item(item, log_entry, i, len(queue),
                         chunk_text=chunk_text if show_chunk else None)
            decision, note = get_decision(has_chunk=bool(chunk_text))
            if decision == "c":
                show_chunk = not show_chunk   # toggle full chunk display
                continue
            break

        if decision == "q":
            break
        if decision == "s":
            skipped += 1
            continue

        entry = {
            "item_id":    item["item_id"],
            "decision":   "accept" if decision == "a" else "reject",
            "note":       note,
            "topic":      item.get("topic", []),
            "confidence": (log_entry or {}).get("confidence", "unknown"),
        }
        results.append(entry)

    # Append to results file (supports multiple sessions)
    with RESULTS_PATH.open("a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    accepted = [r for r in results if r["decision"] == "accept"]
    rejected = [r for r in results if r["decision"] == "reject"]
    rate     = len(accepted) / len(results) * 100 if results else 0

    print(f"\n{'─' * 50}")
    print(f"Session summary: reviewed {len(results)}  skipped {skipped}")
    print(f"  Accepted: {len(accepted)}  Rejected: {len(rejected)}  ({rate:.0f}% acceptance)")

    if rejected:
        print("\nRejected items:")
        for r in rejected:
            note = f"  — {r['note']}" if r["note"] else ""
            print(f"  {r['item_id']}{note}")

    # Per-topic acceptance
    by_topic: dict[str, list[str]] = defaultdict(list)
    for r in results:
        for t in r.get("topic", ["(untagged)"]):
            by_topic[t].append(r["decision"])

    if len(by_topic) > 1:
        print("\nPer-topic acceptance:")
        for topic in sorted(by_topic):
            decisions = by_topic[topic]
            acc = sum(1 for d in decisions if d == "accept")
            print(f"  {topic[:50]:<50}  {acc}/{len(decisions)}")

    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=30,
                        help="Number of items to review (default: 30)")
    parser.add_argument("--all",    action="store_true", dest="review_all",
                        help="Review every generated item")
    parser.add_argument("--resume", action="store_true",
                        help="Skip items already in qc_results.jsonl")
    args = parser.parse_args()
    main(args.n, args.review_all, args.resume)
