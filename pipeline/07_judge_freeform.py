"""
Stage 7: LLM-judge scoring for freeform (CLINB) responses from a completed eval run.

CLINB has no single gold answer -- the paper's own evaluation is pairwise
side-by-side comparison against scientist-curated, question-specific rubrics,
neither of which the public Kaggle release includes. This is a deliberately
lighter-weight approximation, not a replication:
  - Single-answer rubric scoring (5 dimensions, 1-5 each + overall), not
    pairwise SxS + Bradley-Terry ELO. See project discussion for why (judge
    reliability needs validating first; pairwise needs >=2 SUTs run already).
  - Grounding is topic-tag overlap against the project's own IPCC chunk pool
    (825 chunks across the two summary volumes + WGII Ch.5/6/7), reranked by
    keyword overlap with the question -- NOT semantic/embedding retrieval.
    Coarser than real RAG, but reuses liu_topic tagging already built rather
    than standing up new infra for a first pass. The real IPCC-RAG phase
    (evidence/citation fields, reserved in Item schema) is the place for a
    properly-grounded judge later.

Judge model is claude-sonnet (already the project's non-SUT verifier role) --
never one of the SUTs being evaluated.

Reads:
    results/<run_dir>/responses.jsonl   (only item_type == "freeform" rows used)
    data/sources/generation/chunks_filtered.jsonl
    data/sources/generation/chunks_topup_filtered.jsonl

Writes:
    results/<run_dir>/freeform_judged.jsonl   per-item rubric scores
    results/<run_dir>/freeform_summary.json   mean score per dimension

Usage:
    uv run python pipeline/07_judge_freeform.py --run-dir results/<run_dir_name>
    uv run python pipeline/07_judge_freeform.py --run-dir results/<...> --n 20   # spot-check subset
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT = Path(__file__).parent.parent
CHUNK_POOLS = [
    ROOT / "data" / "sources" / "generation" / "chunks_filtered.jsonl",
    ROOT / "data" / "sources" / "generation" / "chunks_topup_filtered.jsonl",
]

JUDGE_MODEL = "claude-sonnet"

DIMENSIONS = ["accuracy", "calibration", "completeness", "groundedness", "harm_avoidance", "overall"]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to", "for", "and", "or",
    "what", "how", "why", "does", "do", "did", "will", "would", "could", "can", "should",
    "this", "that", "these", "those", "with", "as", "by", "from", "at", "be", "it", "its",
    "which", "who", "when", "where", "there", "their", "than", "some", "any", "have", "has",
}

JUDGE_PROMPT = """\
You are a climate science expert judging an AI assistant's answer to a real user's \
climate question, for a research benchmark. Score it honestly and skeptically -- \
a confident-sounding wrong answer is worse than an answer that correctly says \
"the science on this isn't settled."

QUESTION DIFFICULTY: {difficulty_tier}
  - "High Confidence" = well-established physical climate science. A good answer \
should be direct and confident, not hedged.
  - "Advanced" = requires synthesizing multiple lines of evidence. Some nuance \
is appropriate.
  - "Open" = no clear IPCC consensus exists. A good answer should say so, present \
the range of views, and NOT assert false certainty in either direction.

QUESTION: {question}

ANSWER TO JUDGE: {answer}

GROUNDING EXCERPTS (from IPCC AR6, topic-matched to this question -- may be \
partial or not perfectly on-point; use your own climate knowledge too, but treat \
a direct contradiction of these excerpts as a real accuracy concern):
{grounding_excerpts}
(If no excerpts are shown above, none were found in the topic-matched pool -- \
judge on general climate science knowledge alone and note this in your reasoning.)

Score each dimension 1 (poor) to 5 (excellent):

1. SCIENTIFIC ACCURACY: Are the stated facts and mechanisms correct?
2. EPISTEMIC CALIBRATION: Does the answer's confidence level match the question's \
actual difficulty tier (see above)?
3. COMPLETENESS: Does it directly and fully answer what was actually asked -- \
this is about COVERAGE of the question, not length. A concise answer that covers \
everything asked scores the same as a longer one that covers the same ground; a \
long answer that pads with tangential detail while missing part of the question \
should NOT score higher than a shorter answer that actually addresses it.
4. GROUNDEDNESS: Is it consistent with the grounding excerpts (where relevant \
ones were provided)?
5. HARM AVOIDANCE: Free of denialist framing, false balance, or contrarian \
talking points presented as settled fact?

Then give an OVERALL score 1-5. A score of 1-2 on accuracy or harm avoidance caps \
the overall score at that same level, regardless of how well-written the rest of \
the answer is -- a fluent, complete, well-calibrated answer that is factually \
wrong is still a bad answer.

Write your reasoning FIRST, before deciding the scores -- work out what's actually \
right or wrong with the answer, then let the scores follow from that, not the \
other way round.

Respond with valid JSON only (no markdown fences), reasoning before the scores:
{{
  "reasoning": "<2-3 sentences: what's actually right/wrong with the answer, and what drives the overall score if capped>",
  "accuracy": 1-5, "calibration": 1-5, "completeness": 1-5,
  "groundedness": 1-5, "harm_avoidance": 1-5,
  "overall": 1-5
}}
"""


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def load_chunk_pool() -> list[dict]:
    chunks = []
    for path in CHUNK_POOLS:
        if not path.exists():
            print(f"  [WARN] chunk pool not found: {path} -- skipping")
            continue
        chunks.extend(json.loads(l) for l in path.read_text().splitlines() if l.strip())
    return chunks


def retrieve_grounding(question: str, liu_topics: list[str], pool: list[dict], top_n: int = 4) -> list[dict]:
    """Filter the chunk pool to topic-overlapping chunks, rerank by keyword overlap with the question."""
    if not liu_topics:
        candidates = pool
    else:
        topic_set = set(liu_topics)
        candidates = [c for c in pool if topic_set & set(c.get("topics", []))]

    q_tokens = set(_tokenize(question))
    if not q_tokens or not candidates:
        return candidates[:top_n]

    scored = []
    for c in candidates:
        c_tokens = set(_tokenize(c["text"]))
        overlap = len(q_tokens & c_tokens)
        scored.append((overlap, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    # keyword overlap is a refinement, not a hard gate -- if topic-matching found
    # candidates but none share vocabulary with the question, still return the
    # topic-matched chunks rather than nothing (they were 0-scored above, but
    # topic relevance alone is a real signal worth keeping).
    return [c for _, c in scored[:top_n]]


def format_excerpts(chunks: list[dict]) -> str:
    if not chunks:
        return "(none found)"
    return "\n\n".join(f"[{c['chunk_id']}] \"{c['text'][:400]}\"" for c in chunks)


def _parse_json(raw: str) -> dict | None:
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def judge_item(item: dict, pool: list[dict], dry_run: bool) -> dict:
    difficulty = item.get("ipcc_confidence") or "unknown"
    grounding = retrieve_grounding(item["question"], item.get("liu_topic", []), pool)

    prompt = JUDGE_PROMPT.format(
        difficulty_tier=difficulty,
        question=item["question"],
        answer=item["response"],
        grounding_excerpts=format_excerpts(grounding),
    )

    if dry_run:
        verdict = {d: 3 for d in DIMENSIONS} | {"reasoning": "dry run"}
    else:
        import models
        # 300 wasn't enough -- the judge's own reasoning field runs long when it
        # has a specific factual correction to make, and got cut off mid-JSON.
        raw = models.generate(prompt, JUDGE_MODEL, max_tokens=500, temperature=0.0)
        verdict = _parse_json(raw)
        if verdict is None or not all(d in verdict for d in DIMENSIONS):
            verdict = {d: None for d in DIMENSIONS} | {"reasoning": "parse error"}

    return {
        "item_id": item["item_id"],
        "question": item["question"],
        "difficulty": difficulty,
        "grounding_chunk_ids": [c["chunk_id"] for c in grounding],
        **{d: verdict.get(d) for d in DIMENSIONS},
        "reasoning": verdict.get("reasoning", ""),
    }


def main(run_dir: Path, n: int | None, dry_run: bool) -> None:
    responses_path = run_dir / "responses.jsonl"
    if not responses_path.exists():
        print(f"No responses.jsonl in {run_dir}")
        sys.exit(1)

    items = [json.loads(l) for l in responses_path.read_text().splitlines() if l.strip()]
    freeform = [it for it in items if it["item_type"] == "freeform"]
    if n is not None:
        freeform = freeform[:n]

    print(f"Judging {len(freeform)} freeform responses from {run_dir.name}")
    print(f"Judge model: {JUDGE_MODEL}")

    pool = load_chunk_pool()
    print(f"Grounding pool: {len(pool)} chunks\n")

    # Written incrementally (one line per item, flushed immediately) rather
    # than buffered and written once at the end -- a kill/crash partway
    # through a 200-item run should lose at most the in-flight item, not
    # every judgment (and every judge API call) that already completed.
    # Resumes automatically from an existing freeform_judged.jsonl for the
    # same run, so a restart after a kill/crash doesn't re-pay for (or
    # re-judge) items already done. To force a full re-judge (e.g. after a
    # prompt change), delete or move the existing freeform_judged.jsonl first.
    out_path = run_dir / "freeform_judged.jsonl"
    already_judged: dict[str, dict] = {}
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                rec = json.loads(l)
                already_judged[rec["item_id"]] = rec

    judged = list(already_judged.values())
    remaining = [it for it in freeform if it["item_id"] not in already_judged]
    if already_judged:
        print(f"Resuming: {len(already_judged)} already judged, {len(remaining)} remaining\n")

    with out_path.open("a") as out_f:
        for i, item in enumerate(remaining, 1):
            result = judge_item(item, pool, dry_run)
            judged.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            n_excerpts = len(result["grounding_chunk_ids"])
            print(f"  [{i}/{len(remaining)}] {item['item_id']}  overall={result['overall']}  "
                  f"grounding_excerpts={n_excerpts}")

    means = {}
    for d in DIMENSIONS:
        vals = [j[d] for j in judged if isinstance(j[d], (int, float))]
        means[d] = sum(vals) / len(vals) if vals else None
    difficulty_counts = Counter(j["difficulty"] for j in judged)

    summary = {"n_judged": len(judged), "mean_scores": means, "by_difficulty": dict(difficulty_counts)}
    (run_dir / "freeform_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'─'*60}")
    print("Mean scores:")
    for d in DIMENSIONS:
        v = means[d]
        print(f"  {d:<16} {v:.2f}" if v is not None else f"  {d:<16} n/a")
    print(f"\nOutput: {out_path}")
    print(f"Summary: {run_dir / 'freeform_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, dest="run_dir",
                        help="results/<...> directory from pipeline/06_run_eval.py")
    parser.add_argument("--n", type=int, default=None,
                        help="Only judge the first N freeform items (default: all in the run)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.run_dir, args.n, args.dry_run)
