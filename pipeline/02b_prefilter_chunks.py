"""
Stage 2b: Pre-filter chunks before v2 generation.

Reads:
    data/sources/generation/chunks.jsonl  (output of 02a_chunk_pdfs.py)

Writes:
    data/sources/generation/chunks_filtered.jsonl   chunks that pass the screen
    data/sources/generation/prefilter_log.jsonl     per-chunk verdict + reason

Screen: a cheap LLM call asks "does this chunk contain a self-contained,
testable climate fact?" -- discards captions, references, boilerplate, and
glossary/methods text before they waste a generation + verification call.

This targets QC failure mode 4 ("not relevant / bad chunk") from the v2 brief.
Runs against qwen3-235b-generator (already provisioned as a non-SUT pipeline
model) rather than a small SUT model, so screening never overlaps with the
evaluated model panel.

v1 (pipeline/02_generate_mcq.py) does not use this filter and is untouched --
it consumes chunks.jsonl directly.

Usage:
    uv run python pipeline/02b_prefilter_chunks.py
    uv run python pipeline/02b_prefilter_chunks.py --dry-run --n 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT          = Path(__file__).parent.parent
CHUNKS_PATH   = ROOT / "data" / "sources" / "generation" / "chunks.jsonl"
OUTPUT_PATH   = ROOT / "data" / "sources" / "generation" / "chunks_filtered.jsonl"
LOG_PATH      = ROOT / "data" / "sources" / "generation" / "prefilter_log.jsonl"

PREFILTER_MODEL = "qwen3-235b-generator"

PREFILTER_PROMPT = """\
You are screening candidate passages for a climate-science question-writing pipeline.

Passage:
{passage}

Does this passage contain at least one self-contained, testable climate FACT -- a specific \
claim, number, mechanism, or relationship that could become a standalone climate-knowledge \
question with a single correct answer?

Answer NO if the passage is mostly:
- a figure/table caption or reference ("Figure 3.2 shows...", "see Table TS.1")
- a citation list, references, or methods/data-source description
- boilerplate (headers, confidence-level legends, glossary/definition text)
- too vague or general to support one specific factual question

Respond with valid JSON only (no markdown fences):
{{ "has_fact": true/false, "reason": "<one short phrase>" }}
"""


def _parse_json(raw: str) -> dict | None:
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def screen_chunk(chunk: dict, dry_run: bool) -> dict:
    """Returns {has_fact, reason}. Defaults to keeping the chunk on parse failure."""
    if dry_run:
        return {"has_fact": True, "reason": "dry run"}

    import models
    prompt = PREFILTER_PROMPT.format(passage=chunk["text"])
    # temperature=0.0 for a consistent yes/no screen; short output, cheap call
    raw = models.generate(prompt, PREFILTER_MODEL, max_tokens=80, temperature=0.0)
    result = _parse_json(raw)
    if result is None or "has_fact" not in result:
        return {"has_fact": True, "reason": "parse error -- kept to avoid false negatives"}
    return result


def main(n: int | None, dry_run: bool) -> None:
    if not CHUNKS_PATH.exists():
        print(f"Chunks file not found: {CHUNKS_PATH}")
        print("Run pipeline/02a_chunk_pdfs.py first.")
        sys.exit(1)

    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text().splitlines() if l.strip()]
    if n is not None:
        chunks = chunks[:n]

    print(f"Screening {len(chunks)} chunks with {PREFILTER_MODEL}...")

    kept: list[dict] = []
    log_entries: list[dict] = []

    for i, chunk in enumerate(chunks, 1):
        verdict = screen_chunk(chunk, dry_run)
        has_fact = bool(verdict.get("has_fact", True))
        log_entries.append({
            "chunk_id":  chunk["chunk_id"],
            "has_fact":  has_fact,
            "reason":    verdict.get("reason", ""),
        })
        status = "keep" if has_fact else "drop"
        if i % 25 == 0 or i == len(chunks):
            kept_so_far = sum(1 for e in log_entries if e["has_fact"])
            print(f"  {i}/{len(chunks)} screened  ({kept_so_far} kept so far)  last={chunk['chunk_id']} -> {status}")
        if has_fact:
            kept.append(chunk)

    OUTPUT_PATH.write_text("\n".join(json.dumps(c) for c in kept) + "\n")
    LOG_PATH.write_text("\n".join(json.dumps(e) for e in log_entries) + "\n")

    dropped = len(chunks) - len(kept)
    print(f"\n{'─'*60}")
    print(f"Kept:    {len(kept)}/{len(chunks)}")
    print(f"Dropped: {dropped}/{len(chunks)}")
    print(f"\nOutput:   {OUTPUT_PATH}")
    print(f"Log:      {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Only screen the first N chunks (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls; keep every chunk (for testing)")
    args = parser.parse_args()
    main(args.n, args.dry_run)
