"""
Interactive terminal QC review of generated MCQ items.

Items flagged by the automated verifier as low-confidence are shown first,
unless a claude_review_{pool}.jsonl file exists for the pool (an independent
second-opinion pass), in which case likely_reject items are shown first, then
borderline, then likely_accept -- this takes priority over the confidence
ordering since it's a more targeted signal.

Each item shows the question, options (gold marked), supporting excerpt, the
verifier's verdict, and Claude's second-opinion verdict if one exists.

Commands per item:
  a  accept
  r  reject (+ optional one-line note)
  s  skip
  q  quit and save

Results written to: data/sources/generation/qc_results.jsonl (--pool v1, default)
                 or: data/sources/generation/qc_results_v2.jsonl (--pool v2)
  Each line: {item_id, decision, note, topic, confidence}

Already-reviewed items (from prior runs against the same pool) are ALWAYS
skipped -- there is no opt-in flag for this anymore. Earlier versions of this
script only skipped them with --resume, which meant plain reruns re-served
the same deterministic front-of-queue items instead of advancing, silently
losing review coverage across sessions.

After the session, prints per-topic acceptance rates and lists items
that were rejected, for targeted follow-up.

Usage:
    uv run python scripts/qc_sample.py                 # 30-item sample (low-conf first), v1 pool
    uv run python scripts/qc_sample.py --n 50
    uv run python scripts/qc_sample.py --all           # review every item in the pool
    uv run python scripts/qc_sample.py --pool v2       # review the v2 candidate pool instead
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent

POOL_CONFIG = {
    "v1": {
        "items_path":   ROOT / "data" / "selections" / "generated_mcq.jsonl",
        "log_path":     ROOT / "data" / "sources" / "generation" / "generation_log.jsonl",
        "results_path": ROOT / "data" / "sources" / "generation" / "qc_results.jsonl",
        "review_path":  ROOT / "data" / "sources" / "generation" / "claude_review_v1.jsonl",
    },
    "v2": {
        "items_path":   ROOT / "data" / "selections" / "generated_mcq_v2.jsonl",
        "log_path":     ROOT / "data" / "sources" / "generation" / "generation_log_v2.jsonl",
        "results_path": ROOT / "data" / "sources" / "generation" / "qc_results_v2.jsonl",
        "review_path":  ROOT / "data" / "sources" / "generation" / "claude_review_v2.jsonl",
    },
}

VERDICT_RANK = {"likely_reject": 0, "borderline": 1, "likely_accept": 2}
CHUNKS_PATH = ROOT / "data" / "sources" / "generation" / "chunks.jsonl"

SEED = 42


def load_items(items_path: Path) -> list[dict]:
    if not items_path.exists():
        print(f"No generated items found at {items_path}")
        print("Run pipeline/02_generate_mcq.py (--pool v1) or "
              "pipeline/02_generate_mcq_v2.py (--pool v2) first.")
        sys.exit(1)
    return [json.loads(l) for l in items_path.read_text().splitlines() if l.strip()]


def load_chunks() -> dict[str, str]:
    """Return {chunk_id: chunk_text} for looking up full source context."""
    if not CHUNKS_PATH.exists():
        return {}
    return {
        json.loads(l)["chunk_id"]: json.loads(l)["text"]
        for l in CHUNKS_PATH.read_text().splitlines()
        if l.strip()
    }


def load_log(log_path: Path) -> dict[str, dict]:
    """Return {item_id: log_entry} from the pool's generation log."""
    if not log_path.exists():
        return {}
    entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    return {e["item_id"]: e for e in entries}


def load_existing_results(results_path: Path) -> set[str]:
    """Return set of item_ids already reviewed (any decision) for this pool."""
    if not results_path.exists():
        return set()
    return {
        json.loads(l)["item_id"]
        for l in results_path.read_text().splitlines()
        if l.strip()
    }


def load_claude_review(review_path: Path) -> dict[str, dict]:
    """Return {item_id: {verdict, reason}} from an independent second-opinion pass, if one exists."""
    if not review_path.exists():
        return {}
    entries = [json.loads(l) for l in review_path.read_text().splitlines() if l.strip()]
    return {e["item_id"]: e for e in entries}


def sample_items(
    items: list[dict],
    log: dict[str, dict],
    claude_review: dict[str, dict],
    already_reviewed: set[str],
    n: int,
    review_all: bool,
) -> list[dict]:
    """
    Return items to review, worst-first.

    If a Claude second-opinion review exists for the pool, it takes priority:
    likely_reject, then borderline, then likely_accept. Within each group (and
    whenever no review exists), low-confidence verifier passes come first,
    then the rest in a fixed shuffle.
    Always skips items already reviewed in a prior run against this pool.
    """
    remaining = [it for it in items if it["item_id"] not in already_reviewed]

    low_conf  = [it for it in remaining if log.get(it["item_id"], {}).get("confidence") == "low"]
    other     = [it for it in remaining if it not in low_conf]

    rng = random.Random(SEED)
    rng.shuffle(other)

    ordered = low_conf + other

    if claude_review:
        ordered.sort(key=lambda it: VERDICT_RANK.get(
            claude_review.get(it["item_id"], {}).get("verdict"), 2
        ))

    if review_all:
        return ordered
    return ordered[:n]


def display_item(
    item: dict,
    log_entry: dict | None,
    review_entry: dict | None,
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
        conf  = log_entry.get("confidence", "?")
        notes = log_entry.get("notes", "")
        if "gates" in log_entry:
            # v2 log shape: five named gates instead of two booleans
            gates      = log_entry.get("gates", {})
            gates_str  = "  ".join(f"{g.replace('_pass', '')}={'✓' if ok else '✗'}" for g, ok in gates.items())
            failed     = log_entry.get("failed_gates", [])
            failed_str = f"  failed={failed}" if failed else ""
            print(f"  Verifier: {gates_str}  confidence={conf}{failed_str}")
        else:
            entail = "✓" if log_entry.get("entailed") else "✗"
            dist   = "✓" if log_entry.get("distractors_clean") else "✗"
            failed = log_entry.get("failed_checks", [])
            failed_str = f"  failed={failed}" if failed else ""
            print(f"  Verifier: entailed={entail}  distractors={dist}  confidence={conf}{failed_str}")
        if notes:
            print(f"  Notes:    {notes}")
    if review_entry:
        verdict = review_entry.get("verdict", "?").upper()
        reason  = review_entry.get("reason", "")
        print(f"  Claude review: {verdict}")
        if reason:
            print(f"    {reason}")
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


def main(n: int, review_all: bool, pool: str) -> None:
    cfg = POOL_CONFIG[pool]
    items_path, log_path, results_path, review_path = (
        cfg["items_path"], cfg["log_path"], cfg["results_path"], cfg["review_path"]
    )

    items         = load_items(items_path)
    log           = load_log(log_path)
    claude_review = load_claude_review(review_path)
    chunks        = load_chunks()
    already       = load_existing_results(results_path)

    queue = sample_items(items, log, claude_review, already, n, review_all)

    low_conf_count = sum(
        1 for it in queue
        if log.get(it["item_id"], {}).get("confidence") == "low"
    )
    print(f"Generated items: {len(items)}")
    print(f"Queued for review: {len(queue)}  ({low_conf_count} low-confidence, shown first)")
    if claude_review:
        flagged = sum(1 for it in queue if claude_review.get(it["item_id"], {}).get("verdict") != "likely_accept")
        print(f"Claude second-opinion review found for this pool -- {flagged} flagged item(s) shown first")
    if already:
        print(f"Already reviewed (skipping): {len(already)}")
    if not queue:
        print("Nothing left to review.")
        return

    results_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    skipped = 0

    for i, item in enumerate(queue, 1):
        log_entry    = log.get(item["item_id"])
        review_entry = claude_review.get(item["item_id"])
        chunk_id     = item.get("source_doc_id", "")
        chunk_text   = chunks.get(chunk_id)
        show_chunk   = False

        while True:
            display_item(item, log_entry, review_entry, i, len(queue),
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
    with results_path.open("a") as f:
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

    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=30,
                        help="Number of items to review (default: 30)")
    parser.add_argument("--all",    action="store_true", dest="review_all",
                        help="Review every item in the pool")
    parser.add_argument("--pool",   choices=["v1", "v2"], default="v1",
                        help="Which candidate pool to review (default: v1)")
    args = parser.parse_args()
    main(args.n, args.review_all, args.pool)
