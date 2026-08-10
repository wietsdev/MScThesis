"""
Stage 2: Generate MCQ items from IPCC source passages.

Reads:
    data/sources/generation/chunks.jsonl  (output of 02a_chunk_pdfs.py)
    data/selections/climaqa_selection.jsonl etc.  (to compute topic gaps)

Writes:
    data/selections/generated_mcq.jsonl      accepted items (passed verification)
    data/sources/generation/generation_log.jsonl  per-item audit trail

Pipeline per chunk:
  1. Generate: LLM writes question + options + gold + excerpt from passage
  2. Verify:   second LLM call checks gold is passage-entailed and each
               distractor is passage-excluded; items that fail either are dropped
  3. Flag:     items that pass with low confidence are marked for priority QC

Contamination note: IPCC AR6 Summary Volumes are the generation source only.
They must NOT be used as a retrieval corpus during evaluation.

Usage:
    uv run python pipeline/02_generate_mcq.py
    uv run python pipeline/02_generate_mcq.py --model claude-haiku --n 100
    uv run python pipeline/02_generate_mcq.py --dry-run --n 5   (no LLM calls)
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from item_schema import Item

ROOT           = Path(__file__).parent.parent
CHUNKS_PATH    = ROOT / "data" / "sources" / "generation" / "chunks.jsonl"
SELECTIONS_DIR = ROOT / "data" / "selections"
OUTPUT_PATH    = SELECTIONS_DIR / "generated_mcq.jsonl"
LOG_PATH       = ROOT / "data" / "sources" / "generation" / "generation_log.jsonl"

N_GENERATE = 100
SEED       = 42

TARGET_TOPICS = [
    "B1. Biodiversity Loss",
    "C2. Water Resources & Hydrological Impacts",
    "C3. Human Health & Well-being",
    "C4. Social Equity, Vulnerability & Migration",
    "C5. Urban Systems & Infrastructure Impacts",
    "C6. Service & Industry Sector Impacts",
    "D1. Agricultural & Food System Adaptation",
    "D2. Urban Planning, Adaptation & Resilience",
    "D3. Public Health Adaptation",
    "D4. Public Awareness, Communication & Community Engagement",
    "D5. Natural Resource Management & Conservation",
    "E1. Climate Policy, Governance & Finance Mechanism",
    "E2. Energy Transition",
    "E4. Land Use & Ecosystem-based Mitigation",
    "E5. Transport & Building Emissions Reduction",
]

# Existing selection files whose topic coverage counts toward the gap analysis.
EXISTING_SELECTIONS = [
    "climaqa_selection.jsonl",
    "climate_fever_selection.jsonl",
    "pira_selection.jsonl",
]

GENERATION_PROMPT = """\
You are a climate science expert writing a multiple-choice question for a peer-reviewed academic benchmark.

Topic area: {topic}

Using ONLY the passage below, write one factual question with exactly 4 options (a, b, c, d).

STRICT requirements:
1. SELF-CONTAINED: The question must read as a standalone factual question. \
Do NOT write phrases like "according to the passage", "the text states", "based on the excerpt", \
"as described above", "in this document", or any similar reference to a source. \
The question must make sense without knowing there is a passage.
2. UNIQUELY CORRECT: The correct answer must be uniquely identifiable. \
A knowledgeable reader must NOT be able to reasonably defend any other option as also correct. \
If you cannot guarantee this, write a different question.
3. SPECIFIC DISTRACTORS: Each wrong option must be a specific, plausible climate claim \
that is clearly wrong in the context of this passage — ideally a common misconception \
or a value/quantity/mechanism that sounds right but contradicts the passage. \
Do NOT write obviously false, generic, or nonsensical distractors.
4. The correct answer must be directly supported by a verbatim sentence or phrase in the passage.

Respond with valid JSON only (no markdown fences):
{{
  "question": "...",
  "a": "...",
  "b": "...",
  "c": "...",
  "d": "...",
  "gold": "<letter a/b/c/d>",
  "excerpt": "<one or two verbatim sentences from the passage that unambiguously support the correct answer>"
}}

Passage:
{passage}
"""

VERIFICATION_PROMPT = """\
You are a strict quality-control reviewer for a peer-reviewed climate MCQ benchmark.

Evaluate this item on FOUR checks. Fail the item if ANY check fails.

CHECK 1 — SELF-CONTAINED:
Does the question or any option contain phrases that reference a source document?
Fail if you see: "according to", "the text", "the passage", "as described", \
"this document", "the excerpt", "based on the", "as stated", "the report states", \
or any similar phrase that assumes the reader has a text in front of them.

CHECK 2 — ENTAILMENT:
Is the correct answer (option {gold_upper}) unambiguously entailed by the excerpt alone?
A reader who knows only the excerpt must be able to defend this answer without any outside knowledge.

CHECK 3 — UNIQUENESS:
Is there exactly ONE defensibly correct answer among the four options?
If a knowledgeable climate reader could reasonably argue that any other option is also correct \
(even without the excerpt), fail this check.

CHECK 4 — DISTRACTOR QUALITY:
Are the wrong options specific and plausible — not obviously incorrect or generic?
Fail if any distractor is trivially wrong, nonsensical, or so vague that no informed reader would choose it.

Respond with valid JSON only (no markdown fences):
{{
  "entailed": true/false,
  "distractors_clean": true/false,
  "confidence": "high" | "medium" | "low",
  "failed_checks": ["<list any of: self_contained, entailment, uniqueness, distractor_quality>"],
  "notes": "<one sentence: what specifically failed, or 'all checks passed' if none>"
}}

Question: {question}
Options:
  A) {a}
  B) {b}
  C) {c}
  D) {d}
Correct answer: {gold_upper}) {gold_text}
Excerpt: "{excerpt}"
"""


# ---------------------------------------------------------------------------
# Question-level topic re-tagger
# ---------------------------------------------------------------------------
# Mirrors the keyword table in 02a_chunk_pdfs.py.  Used to tag the generated
# question text rather than inheriting the (potentially broader) chunk tags.

_QUESTION_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "B1. Biodiversity Loss": ["biodiversity", "species loss", "extinction", "habitat loss", "coral reef", "species richness", "endemic species", "range shift", "phenology"],
    "C2. Water Resources & Hydrological Impacts": ["water resource", "groundwater", "water scarcity", "water stress", "hydrological", "water supply", "water security", "water availability", "aquifer", "streamflow"],
    "C3. Human Health & Well-being": ["human health", "disease burden", "mortality", "heat stress", "air quality", "malaria", "dengue", "vector-borne", "mental health", "morbidity", "heat-related illness", "respiratory disease"],
    "C4. Social Equity, Vulnerability & Migration": ["social equity", "vulnerability", "climate migration", "displacement", "climate justice", "inequality", "marginalised", "marginalized", "poverty", "indigenous peoples", "climate refugee", "gender equity"],
    "C5. Urban Systems & Infrastructure Impacts": ["urban system", "city infrastructure", "urban flood", "urban heat island", "critical infrastructure", "transport network", "power grid", "built environment"],
    "C6. Service & Industry Sector Impacts": ["tourism industry", "insurance sector", "financial risk", "supply chain disruption", "economic loss", "economic sector", "service industry"],
    "D1. Agricultural & Food System Adaptation": ["agricultural adaptation", "crop adaptation", "food system adaptation", "agroforestry", "drought-resistant", "heat-tolerant crop", "irrigation efficiency", "precision agriculture", "climate-smart agriculture"],
    "D2. Urban Planning, Adaptation & Resilience": ["urban adaptation", "urban resilience", "green infrastructure", "flood-resilient", "nature-based solution", "adaptation planning", "urban regeneration", "urban climate resilience"],
    "D3. Public Health Adaptation": ["health adaptation", "disease surveillance", "heat action plan", "public health adaptation", "health system strengthening", "vector control", "climate-health adaptation"],
    "D4. Public Awareness, Communication & Community Engagement": ["climate communication", "public awareness", "climate education", "community engagement", "citizen science", "climate literacy", "stakeholder engagement"],
    "D5. Natural Resource Management & Conservation": ["natural resource management", "conservation", "protected area", "ecosystem-based adaptation", "sustainable land management", "forest management", "biodiversity conservation strategy"],
    "E1. Climate Policy, Governance & Finance Mechanism": ["climate policy", "climate governance", "paris agreement", "unfccc", "nationally determined contribution", "ndc", "climate finance", "carbon tax", "carbon pricing", "adaptation finance", "climate legislation"],
    "E2. Energy Transition": ["renewable energy", "solar power", "wind energy", "energy transition", "decarbonisation", "decarbonization", "net zero", "clean energy", "fossil fuel phase", "coal phase-out", "nuclear power", "hydrogen energy", "battery storage"],
    "E4. Land Use & Ecosystem-based Mitigation": ["land use change", "redd", "reforestation", "afforestation", "nature-based solution for mitigation", "soil carbon sequestration", "beccs", "blue carbon", "ecosystem-based mitigation"],
    "E5. Transport & Building Emissions Reduction": ["transport emissions", "electric vehicle", "ev adoption", "building efficiency", "aviation emissions", "shipping emissions", "low-carbon transport", "building retrofit", "sustainable mobility"],
}


def _retag_question(data: dict, fallback_topic: str) -> list[str]:
    """
    Tag a generated question on its own text (question + options + excerpt).
    If no keywords match, return the generation-target topic as fallback so the
    item is never left untagged.
    """
    text = " ".join([
        data.get("question", ""),
        data.get("a", ""), data.get("b", ""),
        data.get("c", ""), data.get("d", ""),
        data.get("excerpt", ""),
    ]).lower()
    matched = [t for t, kws in _QUESTION_TOPIC_KEYWORDS.items() if any(k in text for k in kws)]
    return matched if matched else [fallback_topic]


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def load_current_topic_counts() -> Counter:
    counts: Counter = Counter()
    for fname in EXISTING_SELECTIONS:
        path = SELECTIONS_DIR / fname
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            for t in item.get("topic", []):
                counts[t] += 1
    return counts


def compute_allocation(
    current_counts: Counter,
    n_total: int,
    chunks_by_topic: dict[str, list] | None = None,
) -> dict[str, int]:
    """
    Allocate generation slots per target topic.

    Strategy:
      - Each topic has a desired post-generation count of TARGET_COVERAGE.
      - Raw slot = gap from TARGET_COVERAGE, capped at MAX_PER_TOPIC, floored at 1.
      - If chunk counts are provided, also cap by number of available chunks.
      - Scale the resulting raw allocations to exactly n_total.
    """
    TARGET_COVERAGE = 12  # desired items per topic after generation
    MAX_PER_TOPIC   = 15  # hard cap: no single topic consumes all the budget
    MIN_PER_TOPIC   = 1

    raw: dict[str, int] = {}
    for t in TARGET_TOPICS:
        gap    = max(0, TARGET_COVERAGE - current_counts.get(t, 0))
        alloc  = max(MIN_PER_TOPIC, min(MAX_PER_TOPIC, gap))
        if chunks_by_topic is not None:
            avail = len(chunks_by_topic.get(t, []))
            alloc = min(alloc, avail)
        raw[t] = alloc

    total_raw = sum(raw.values()) or 1
    scale = n_total / total_raw
    scaled = {t: max(MIN_PER_TOPIC, round(v * scale)) for t, v in raw.items()}

    # re-apply chunk cap after scaling (scaling can inflate past available chunks)
    if chunks_by_topic is not None:
        for t in scaled:
            avail = len(chunks_by_topic.get(t, []))
            scaled[t] = min(scaled[t], avail)

    # fix rounding drift by adjusting topics with most headroom first
    diff = n_total - sum(scaled.values())
    headroom = lambda t: (  # noqa: E731
        (len(chunks_by_topic.get(t, [])) if chunks_by_topic else 999) - scaled[t]
    )
    for t in sorted(TARGET_TOPICS, key=headroom, reverse=True):
        if diff == 0:
            break
        adj = 1 if diff > 0 else -1
        if scaled[t] + adj >= MIN_PER_TOPIC:
            if adj > 0 and chunks_by_topic and scaled[t] >= len(chunks_by_topic.get(t, [])):
                continue  # can't add more than available chunks
            scaled[t] += adj
            diff -= adj

    return scaled


# ---------------------------------------------------------------------------
# Generation and verification
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict | None:
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def generate_from_chunk(
    chunk: dict,
    topic: str,
    model_id: str,
    dry_run: bool,
) -> dict | None:
    """Generate MCQ JSON from a passage chunk. Returns parsed dict or None."""
    if dry_run:
        return {
            "question": f"[DRY RUN] Sample question about {topic}?",
            "a": "Option A", "b": "Option B", "c": "Option C", "d": "Option D",
            "gold": "a",
            "excerpt": chunk["text"][:120],
        }

    import models
    prompt = GENERATION_PROMPT.format(topic=topic, passage=chunk["text"])
    # temperature=0.5 for distractor variety; max_tokens=800 for full JSON output
    raw = models.generate(prompt, model_id, max_tokens=800, temperature=0.5)
    return _parse_json(raw)


def verify_item(
    data: dict,
    verify_model_id: str,
    dry_run: bool,
) -> dict:
    """
    Second model call (different model from generator) to check:
      - entailment: correct answer is unambiguously supported by excerpt
      - distractors_clean: each wrong option is clearly excluded by excerpt
    Returns {entailed, distractors_clean, confidence, notes}.
    """
    if dry_run:
        return {"entailed": True, "distractors_clean": True, "confidence": "high", "notes": "dry run"}

    gold_letter = data["gold"].lower()
    gold_text   = data[gold_letter]
    prompt = VERIFICATION_PROMPT.format(
        question=data["question"],
        a=data["a"], b=data["b"], c=data["c"], d=data["d"],
        gold_upper=gold_letter.upper(),
        gold_text=gold_text,
        excerpt=data.get("excerpt", ""),
    )
    import models
    # temperature=0.0 for consistent verdicts; 256 tokens enough for structured JSON
    raw = models.generate(prompt, verify_model_id, max_tokens=256, temperature=0.0)
    result = _parse_json(raw)
    if result is None or "entailed" not in result:
        return {"entailed": False, "distractors_clean": False, "confidence": "low", "notes": "parse error"}
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    model_id: str,
    verify_model_id: str,
    n_generate: int,
    dry_run: bool,
    seed: int,
) -> None:
    if not CHUNKS_PATH.exists():
        print(f"Chunks file not found: {CHUNKS_PATH}")
        print("Run pipeline/02a_chunk_pdfs.py first.")
        sys.exit(1)

    # Load chunks grouped by topic
    all_chunks = [json.loads(l) for l in CHUNKS_PATH.read_text().splitlines() if l.strip()]
    chunks_by_topic: dict[str, list[dict]] = defaultdict(list)
    for chunk in all_chunks:
        for t in chunk["topics"]:
            chunks_by_topic[t].append(chunk)

    print(f"Loaded {len(all_chunks)} chunks.")

    # Gap analysis
    current_counts = load_current_topic_counts()
    allocation     = compute_allocation(current_counts, n_generate, chunks_by_topic)

    print("\nGap analysis (target topic → current items → slots to generate):")
    for topic in TARGET_TOPICS:
        current = current_counts.get(topic, 0)
        slots   = allocation.get(topic, 0)
        avail   = len(chunks_by_topic.get(topic, []))
        print(f"  {topic[:48]:<48}  cur={current:3d}  gen={slots:3d}  chunks={avail:4d}")

    # Generation loop
    rng = random.Random(seed)
    items:      list[Item] = []
    log_entries: list[dict] = []

    print(f"\nGeneration model:    {model_id}")
    print(f"Verification model:  {verify_model_id}")

    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    for topic in TARGET_TOPICS:
        slots  = allocation.get(topic, 0)
        pool   = list(chunks_by_topic.get(topic, []))
        if not pool:
            print(f"\n  [WARN] No chunks for {topic} — skipping")
            continue

        rng.shuffle(pool)
        accepted = 0
        chunk_idx = 0

        print(f"\n  {topic[:50]}")
        while accepted < slots and chunk_idx < len(pool):
            chunk = pool[chunk_idx]
            chunk_idx += 1

            data = generate_from_chunk(chunk, topic, model_id, dry_run)
            if data is None:
                print("    gen parse error — skipping chunk")
                continue

            required = {"question", "a", "b", "c", "d", "gold", "excerpt"}
            if not required.issubset(data):
                print("    missing fields — skipping chunk")
                continue

            gold = data["gold"].strip().lower()
            if gold not in {"a", "b", "c", "d"}:
                print("    invalid gold — skipping chunk")
                continue

            # Verification pass (different model from generator)
            verification = verify_item(data, verify_model_id, dry_run)
            passed = verification.get("entailed", False) and verification.get("distractors_clean", False)
            confidence = verification.get("confidence", "low")

            log_entry = {
                "item_id":           f"generated_mcq_{chunk['chunk_id']}",
                "chunk_id":          chunk["chunk_id"],
                "topic":             topic,
                "passed":            passed,
                "entailed":          verification.get("entailed"),
                "distractors_clean": verification.get("distractors_clean"),
                "confidence":        confidence,
                "failed_checks":     verification.get("failed_checks", []),
                "notes":             verification.get("notes", ""),
            }
            log_entries.append(log_entry)

            status = f"✓ [{confidence}]" if passed else "✗ (failed verification)"
            print(f"    chunk {chunk['chunk_id']} → {status}")

            if not passed:
                continue

            item = Item(
                item_id        = f"generated_mcq_{chunk['chunk_id']}",
                source         = "generated_mcq",
                license        = "CC-BY-4.0",  # IPCC AR6 is CC-BY-4.0-IGO
                item_type      = "mcq",
                language       = "en",
                question       = data["question"],
                gold           = gold,
                options        = {k: data[k] for k in ("a", "b", "c", "d")},
                # Re-tag topic on the question text itself (not the chunk) so that
                # multi-topic chunks don't inherit unrelated tags. Fall back to the
                # generation target topic if no keywords match the question.
                topic          = _retag_question(data, topic),
                source_doc_id  = chunk["chunk_id"],
                source_excerpt = data.get("excerpt"),
                # mark low-confidence passes for priority QC
                complexity     = "REVIEW" if confidence == "low" else None,
            )
            items.append(item)
            accepted += 1

        if chunk_idx >= len(pool) and accepted < slots:
            print(f"    [WARN] exhausted {len(pool)} chunks, got {accepted}/{slots}")

    # Write outputs
    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(it.to_dict()) for it in items) + "\n"
    )
    LOG_PATH.write_text(
        "\n".join(json.dumps(e) for e in log_entries) + "\n"
    )

    passed  = sum(1 for e in log_entries if e["passed"])
    failed  = sum(1 for e in log_entries if not e["passed"])
    flagged = sum(1 for it in items if it.complexity == "REVIEW")

    print(f"\n{'─'*60}")
    print(f"Generated:  {len(items)} items accepted")
    print(f"  Verified: {passed} passed  |  {failed} dropped by verifier")
    print(f"  Flagged for priority QC (low confidence): {flagged}")
    print(f"\nOutput:     {OUTPUT_PATH}")
    print(f"Audit log:  {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="qwen3-235b-generator",
        help="Gateway model ID for generation (default: qwen3-235b-generator)",
    )
    parser.add_argument(
        "--verify-model", default="claude-sonnet",
        dest="verify_model",
        help="Gateway model ID for verification pass (default: claude-sonnet)",
    )
    parser.add_argument("--n",       type=int, default=N_GENERATE,
                        help="Total items to generate (default: 100)")
    parser.add_argument("--seed",    type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls; write placeholder items for testing")
    args = parser.parse_args()
    main(args.model, args.verify_model, args.n, args.dry_run, args.seed)
