"""
Stage 2 (v2): Generate MCQ items with the reframed prompt + tightened verifier.

This is a PARALLEL pipeline to pipeline/02_generate_mcq.py (v1). It does not
read, write, or import anything from v1 -- v1's prompts, its
generated_mcq.jsonl, and its generation_log.jsonl are untouched and remain a
usable fallback.

Reads:
    data/sources/generation/chunks_filtered.jsonl  (output of 02b_prefilter_chunks.py --
                                                     run that first)
    data/selections/climaqa_selection.jsonl etc.   (existing topic coverage)
    data/selections/accepted_v1.jsonl              (already-accepted v1 items, if present --
                                                     counted toward topic coverage so v2 doesn't
                                                     over-generate for topics v1 already covers)

Writes:
    data/selections/generated_mcq_v2.jsonl         accepted items (passed v2 verifier)
    data/sources/generation/generation_log_v2.jsonl  per-item audit trail, incl. per-gate
                                                       verdicts and distractor justifications

What changed vs v1 (see brief for full rationale):
  1. Generation prompt reframed: "a real person asks a climate assistant a question whose
     answer happens to be in this passage" instead of "write a question about this passage."
     Targets QC failure mode 1 (questions that reference "the text"/"the passage").
  2. Distractor rule tightened: each wrong option must be plausible to a non-expert (a real
     misconception, a fact true elsewhere but wrong here, a plausible wrong number) with a
     one-line justification stored for QC. Targets failure mode 2 (throwaway distractors).
  3. Verifier split into 5 independent gates (self-contained, entailment, uniqueness,
     distractor plausibility, tests-knowledge-not-reading) instead of 2 booleans. The script
     computes passed = all(gates) itself rather than trusting the model's self-reported
     verdict, and logs any disagreement as a signal of verifier confusion. Targets failure
     mode 3 (verifier too lenient on "not entailed / multiple correct").
  4. Chunk pre-filtering (02b) happens upstream -- this script only ever sees chunks that
     already passed the fact-bearing screen. Targets failure mode 4 (bad seed chunks).

Usage:
    uv run python pipeline/02_generate_mcq_v2.py
    uv run python pipeline/02_generate_mcq_v2.py --surplus-total 30   # small QC batch first
    uv run python pipeline/02_generate_mcq_v2.py --dry-run --n-new 5
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from item_schema import Item

ROOT              = Path(__file__).parent.parent
CHUNKS_PATH       = ROOT / "data" / "sources" / "generation" / "chunks_filtered.jsonl"
SELECTIONS_DIR    = ROOT / "data" / "selections"
ACCEPTED_V1_PATH  = SELECTIONS_DIR / "accepted_v1.jsonl"
OUTPUT_PATH       = SELECTIONS_DIR / "generated_mcq_v2.jsonl"
LOG_PATH          = ROOT / "data" / "sources" / "generation" / "generation_log_v2.jsonl"

SURPLUS_TOTAL_DEFAULT = 180   # total candidate pool desired: accepted_v1 + v2 candidates
SEED = 42

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
# accepted_v1.jsonl is added separately (see load_current_topic_counts) since it
# may not exist yet on a first run.
EXISTING_SELECTIONS = [
    "climaqa_selection.jsonl",
    "climate_fever_selection.jsonl",
    "pira_selection.jsonl",
]

GENERATION_PROMPT_V2 = """\
You are building a benchmark of climate knowledge questions used to test AI assistants.

Topic area: {topic}

Picture a curious, moderately climate-literate person chatting with a climate information \
assistant. They ask a genuine question about {topic} -- a question about the WORLD, not \
about a document. Write ONE question they would plausibly type into that chat, whose \
correct answer happens to be a fact stated in the passage below.

The passage's only job is to prove the correct answer is true. It must never leak into the \
question itself.

HARD RULES -- breaking any one makes the item invalid:
- The question must be answerable by a climate-literate person who has NEVER seen the passage.
- NEVER write "according to the passage/text/report/study", "the authors state", "as shown \
above", "in this document", "based on the excerpt", or any phrase assuming a source document \
is in front of the reader.
- The question must read like something a real user would type into a chat window, not like \
a reading-comprehension exercise about this specific text.
- The correct answer is a fact true in the real world; the passage is only your proof of it, \
never the subject of the question.
- The passage may contain inline citations (author names, years, e.g. "(Watts et al., 2019)") \
-- these are NOT part of the fact. Strip them entirely. Never let an author name, year, or \
citation marker appear anywhere in the question, options, or excerpt you write, even if the \
source passage contains one right next to the fact you are using.

DISTRACTOR RULES:
- Each of the three wrong options must be PLAUSIBLE to a non-expert: a common climate \
misconception, a real climate fact that is true elsewhere but wrong in THIS context, a \
plausible-but-wrong number, or a real contrarian talking point.
- Never write an option that is obviously silly, extreme, or nonsensical -- every option \
should look like something a person could genuinely believe.
- All four options must be similar in length, specificity, and grammatical form. The correct \
answer must not be noticeably longer, more detailed, or more hedged than the distractors.

For each wrong option, add a one-sentence justification: why a non-expert might plausibly \
believe it, and specifically why it's wrong here. This is for QC only -- never shown to \
whoever is being tested.

Respond with valid JSON only (no markdown fences):
{{
  "question": "...",
  "a": "...", "b": "...", "c": "...", "d": "...",
  "gold": "<letter a/b/c/d>",
  "excerpt": "<verbatim sentence(s) from the passage that prove the correct answer is true>",
  "distractor_justifications": {{
    "<letter>": "<one sentence: why plausible, why wrong>"
  }}
}}

Passage:
{passage}
"""

VERIFICATION_PROMPT_V2 = """\
You are a strict quality-control reviewer for a peer-reviewed climate MCQ benchmark. You are \
looking for reasons to REJECT -- a good item must clearly pass every gate below.

Check each gate independently, true/false. If ANY gate is false, the item is REJECTed \
overall, regardless of the others.

GATE 1 -- NO SOURCE REFERENCE:
Does the question or any option reference a passage, text, study, report, author, or figure \
in any way ("according to", "the text states", "this study", "the authors", "as shown", \
"in this document")? gate1_pass = false if yes.

GATE 2 -- ENTAILED:
Is the correct answer (option {gold_upper}) entailed by the excerpt ALONE, with no outside \
knowledge needed to defend it? gate2_pass = false if the excerpt doesn't fully support it.

GATE 3 -- SINGLE DEFENSIBLE ANSWER:
Could a knowledgeable climate reader reasonably defend any option OTHER than {gold_upper} as \
also correct (with or without the excerpt)? gate3_pass = false if yes.

GATE 4 -- DISTRACTORS PLAUSIBLE:
Are all three wrong options plausible to a non-expert (a real misconception, a real fact \
that's wrong in this context, a plausible wrong number) rather than obviously/trivially \
wrong? gate4_pass = false if any distractor is a throwaway or absurd option.

GATE 5 -- TESTS KNOWLEDGE, NOT READING:
Is this fundamentally a climate-knowledge question -- would it make sense in a quiz with no \
passage attached -- rather than a reading-comprehension question about this specific text? \
gate5_pass = false if it's really testing whether the reader can locate a sentence in a \
document.

Respond with valid JSON only (no markdown fences):
{{
  "gate1_pass": true/false, "gate2_pass": true/false, "gate3_pass": true/false,
  "gate4_pass": true/false, "gate5_pass": true/false,
  "overall": "accept" | "reject",
  "confidence": "high" | "medium" | "low",
  "failed_gates": ["<gate numbers that failed, e.g. 1, 3>"],
  "notes": "<one sentence: what specifically failed, or 'all gates passed'>"
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

GATE_NAMES = [
    "gate1_pass", "gate2_pass", "gate3_pass", "gate4_pass", "gate5_pass",
]


# ---------------------------------------------------------------------------
# Question-level topic re-tagger (mirrors 02_generate_mcq.py / 02a_chunk_pdfs.py)
# ---------------------------------------------------------------------------

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
    Tag on the question's own text, but ALWAYS keep the generation target
    topic too. A prior version returned matched-keywords-OR-fallback (either/or),
    which silently dropped the generation target whenever a wrong-option's
    wording happened to match a keyword for a completely different topic --
    e.g. an option containing the phrase "public awareness about climate
    change adaptation" tagged a D3-generated item as D4-only, undercounting
    D3 even though the item was entailed by, and generated for, a D3 chunk.
    """
    text = " ".join([
        data.get("question", ""),
        data.get("a", ""), data.get("b", ""),
        data.get("c", ""), data.get("d", ""),
        data.get("excerpt", ""),
    ]).lower()
    matched = [t for t, kws in _QUESTION_TOPIC_KEYWORDS.items() if any(k in text for k in kws)]
    if fallback_topic not in matched:
        matched.append(fallback_topic)
    return matched


# ---------------------------------------------------------------------------
# Gap analysis (same shape as v1's compute_allocation, but current_counts also
# includes accepted_v1 items so v2 doesn't re-generate for topics v1 already
# covers well)
# ---------------------------------------------------------------------------

ACCEPTED_V2_PATH = SELECTIONS_DIR / "accepted_v2.jsonl"
# PIRA/CLINB carry no native Liu tags (see src/item_schema.py) -- their Liu
# coverage lives in liu_topic, backfilled by pipeline/05_tag_topics.py.
LIU_TOPIC_FILES = ["pira_selection.jsonl", "clinb_selection.jsonl"]


def load_current_topic_counts() -> Counter:
    """
    Combined topic coverage across the WHOLE benchmark, not just the sources
    v1 originally targeted -- a prior version of this function only counted
    EXISTING_SELECTIONS + accepted_v1, which meant a second generation round
    didn't know accepted_v2 (or PIRA/CLINB, once tagged) already covered a
    topic, and would over-allocate slots to already-satisfied topics.
    """
    counts: Counter = Counter()
    native_files = list(EXISTING_SELECTIONS)
    for fname, path in [("accepted_v1.jsonl", ACCEPTED_V1_PATH), ("accepted_v2.jsonl", ACCEPTED_V2_PATH)]:
        if path.exists():
            native_files.append(path.name)
        else:
            print(f"  [note] {fname} not found -- won't count toward topic coverage.")

    for fname in native_files:
        path = SELECTIONS_DIR / fname
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            for t in item.get("topic", []):
                counts[t] += 1

    for fname in LIU_TOPIC_FILES:
        path = SELECTIONS_DIR / fname
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            for t in item.get("liu_topic", []):
                counts[t] += 1

    return counts


def compute_allocation(
    current_counts: Counter,
    n_total: int,
    chunks_by_topic: dict[str, list] | None = None,
    topics: list[str] | None = None,
) -> dict[str, int]:
    """
    Allocate v2 generation slots per target topic. Same algorithm as v1's
    compute_allocation: gap-from-target-coverage, capped, then scaled to n_total.
    `topics` restricts the allocation to a subset (e.g. a scoped top-up round);
    defaults to all of TARGET_TOPICS.
    """
    topics = topics if topics is not None else TARGET_TOPICS
    TARGET_COVERAGE = 12
    MAX_PER_TOPIC   = 15
    MIN_PER_TOPIC   = 1

    raw: dict[str, int] = {}
    for t in topics:
        gap    = max(0, TARGET_COVERAGE - current_counts.get(t, 0))
        alloc  = max(MIN_PER_TOPIC, min(MAX_PER_TOPIC, gap))
        if chunks_by_topic is not None:
            avail = len(chunks_by_topic.get(t, []))
            alloc = min(alloc, avail)
        raw[t] = alloc

    total_raw = sum(raw.values()) or 1
    scale = n_total / total_raw
    scaled = {t: max(MIN_PER_TOPIC, round(v * scale)) for t, v in raw.items()}

    if chunks_by_topic is not None:
        for t in scaled:
            avail = len(chunks_by_topic.get(t, []))
            scaled[t] = min(scaled[t], avail)

    diff = n_total - sum(scaled.values())
    headroom = lambda t: (  # noqa: E731
        (len(chunks_by_topic.get(t, [])) if chunks_by_topic else 999) - scaled[t]
    )
    for t in sorted(topics, key=headroom, reverse=True):
        if diff == 0:
            break
        adj = 1 if diff > 0 else -1
        if scaled[t] + adj >= MIN_PER_TOPIC:
            if adj > 0 and chunks_by_topic and scaled[t] >= len(chunks_by_topic.get(t, [])):
                continue
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
    if dry_run:
        return {
            "question": f"[DRY RUN v2] Sample question about {topic}?",
            "a": "Option A", "b": "Option B", "c": "Option C", "d": "Option D",
            "gold": "a",
            "excerpt": chunk["text"][:120],
            "distractor_justifications": {"b": "dry run", "c": "dry run", "d": "dry run"},
        }

    import models
    prompt = GENERATION_PROMPT_V2.format(topic=topic, passage=chunk["text"])
    raw = models.generate(prompt, model_id, max_tokens=900, temperature=0.5)
    return _parse_json(raw)


def verify_item(
    data: dict,
    verify_model_id: str,
    dry_run: bool,
) -> dict:
    """
    Returns per-gate booleans + confidence + notes. passed is computed by the
    caller as all(gate) -- the model's own "overall" field is kept only to
    detect disagreement (logged, not trusted).
    """
    if dry_run:
        return {g: True for g in GATE_NAMES} | {
            "overall": "accept", "confidence": "high", "failed_gates": [], "notes": "dry run",
        }

    gold_letter = data["gold"].lower()
    gold_text   = data[gold_letter]
    prompt = VERIFICATION_PROMPT_V2.format(
        question=data["question"],
        a=data["a"], b=data["b"], c=data["c"], d=data["d"],
        gold_upper=gold_letter.upper(),
        gold_text=gold_text,
        excerpt=data.get("excerpt", ""),
    )
    import models
    raw = models.generate(prompt, verify_model_id, max_tokens=350, temperature=0.0)
    result = _parse_json(raw)
    if result is None or not all(g in result for g in GATE_NAMES):
        return {g: False for g in GATE_NAMES} | {
            "overall": "reject", "confidence": "low", "failed_gates": ["parse_error"],
            "notes": "parse error",
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    model_id: str,
    verify_model_id: str,
    surplus_total: int,
    n_new: int | None,
    dry_run: bool,
    seed: int,
    chunks_path: Path = CHUNKS_PATH,
    output_path: Path = OUTPUT_PATH,
    log_path: Path = LOG_PATH,
    topics: list[str] | None = None,
) -> None:
    if not chunks_path.exists():
        print(f"Filtered chunks file not found: {chunks_path}")
        print("Run pipeline/02b_prefilter_chunks.py first.")
        sys.exit(1)

    # `topics` (given in priority order for a scoped top-up round) restricts
    # which topics this run touches AND determines ownership priority below --
    # defaults to TARGET_TOPICS order when not overridden.
    effective_topics = topics if topics is not None else TARGET_TOPICS

    all_chunks = [json.loads(l) for l in chunks_path.read_text().splitlines() if l.strip()]

    # A chunk tagged with multiple topics is assigned to exactly ONE owning
    # topic here, fixed by effective_topics order alone -- never by slot
    # counts. item_id is derived from chunk_id, so which topic generates a
    # chunk must be stable across runs regardless of --n-new/--surplus-total,
    # or the same item_id silently gets different content between a small
    # test run and a full-scale run, invalidating any manual QC already done
    # on it. For a top-up round, listing the scarce topic first (e.g. D3
    # before C3) ensures a multi-tagged chunk goes to the topic that needs it.
    chunks_by_topic: dict[str, list[dict]] = defaultdict(list)
    for chunk in all_chunks:
        owner = next((t for t in effective_topics if t in chunk["topics"]), None)
        if owner is not None:
            chunks_by_topic[owner].append(chunk)

    print(f"Loaded {len(all_chunks)} pre-filtered chunks from {chunks_path.name}.")
    if topics is not None:
        print(f"Scoped to {len(effective_topics)} topics (priority order): {', '.join(effective_topics)}")

    n_accepted_v1 = 0
    if ACCEPTED_V1_PATH.exists():
        n_accepted_v1 = sum(1 for l in ACCEPTED_V1_PATH.read_text().splitlines() if l.strip())

    if n_new is None:
        n_new = max(1, surplus_total - n_accepted_v1)
        print(f"Surplus target: {surplus_total} total  ({n_accepted_v1} already accepted in "
              f"v1  ->  generating {n_new} new v2 candidates)")
    else:
        print(f"Generating {n_new} new v2 candidates (--n-new override; "
              f"{n_accepted_v1} v1-accepted items are separate)")

    current_counts = load_current_topic_counts()
    allocation     = compute_allocation(current_counts, n_new, chunks_by_topic, topics=effective_topics)

    print("\nGap analysis (target topic -> current items across the whole benchmark -> v2 slots):")
    for topic in effective_topics:
        current = current_counts.get(topic, 0)
        slots   = allocation.get(topic, 0)
        avail   = len(chunks_by_topic.get(topic, []))
        print(f"  {topic[:48]:<48}  cur={current:3d}  gen={slots:3d}  chunks={avail:4d}")

    rng = random.Random(seed)
    items:       list[Item] = []
    log_entries: list[dict] = []

    print(f"\nGeneration model:    {model_id}")
    print(f"Verification model:  {verify_model_id}")

    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for topic in effective_topics:
        slots = allocation.get(topic, 0)
        pool  = list(chunks_by_topic.get(topic, []))
        if not pool:
            print(f"\n  [WARN] No filtered chunks for {topic} -- skipping")
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
                print("    gen parse error -- skipping chunk")
                continue

            required = {"question", "a", "b", "c", "d", "gold", "excerpt"}
            if not required.issubset(data):
                print("    missing fields -- skipping chunk")
                continue

            gold = data["gold"].strip().lower()
            if gold not in {"a", "b", "c", "d"}:
                print("    invalid gold -- skipping chunk")
                continue

            verification = verify_item(data, verify_model_id, dry_run)
            gate_results = [bool(verification.get(g, False)) for g in GATE_NAMES]
            passed       = all(gate_results)
            confidence   = verification.get("confidence", "low")

            model_overall = verification.get("overall", "")
            overall_disagrees = (
                (model_overall == "accept" and not passed) or
                (model_overall == "reject" and passed)
            )

            log_entry = {
                "item_id":                 f"generated_mcq_v2_{chunk['chunk_id']}",
                "chunk_id":                chunk["chunk_id"],
                "topic":                   topic,
                "passed":                  passed,
                "gates":                   dict(zip(GATE_NAMES, gate_results)),
                "confidence":              confidence,
                "failed_gates":            verification.get("failed_gates", []),
                "notes":                   verification.get("notes", ""),
                "model_overall":           model_overall,
                "overall_disagrees_with_gates": overall_disagrees,
                "distractor_justifications": data.get("distractor_justifications", {}),
            }
            log_entries.append(log_entry)

            status = f"✓ [{confidence}]" if passed else "✗ (failed gate check)"
            flag = "  [MODEL/GATE DISAGREE]" if overall_disagrees else ""
            print(f"    chunk {chunk['chunk_id']} -> {status}{flag}")

            if not passed:
                continue

            item = Item(
                item_id        = f"generated_mcq_v2_{chunk['chunk_id']}",
                source         = "generated_mcq",
                license        = "CC-BY-4.0",
                item_type      = "mcq",
                language       = "en",
                question       = data["question"],
                gold           = gold,
                options        = {k: data[k] for k in ("a", "b", "c", "d")},
                topic          = _retag_question(data, topic),
                source_doc_id  = chunk["chunk_id"],
                source_excerpt = data.get("excerpt"),
                complexity     = "REVIEW" if confidence == "low" else None,
            )
            items.append(item)
            accepted += 1

        if chunk_idx >= len(pool) and accepted < slots:
            print(f"    [WARN] exhausted {len(pool)} filtered chunks, got {accepted}/{slots}")

    output_path.write_text(
        "\n".join(json.dumps(it.to_dict()) for it in items) + "\n"
    )
    log_path.write_text(
        "\n".join(json.dumps(e) for e in log_entries) + "\n"
    )

    passed_n  = sum(1 for e in log_entries if e["passed"])
    failed_n  = sum(1 for e in log_entries if not e["passed"])
    flagged   = sum(1 for it in items if it.complexity == "REVIEW")
    disagreed = sum(1 for e in log_entries if e["overall_disagrees_with_gates"])

    print(f"\n{'─'*60}")
    print(f"Generated:  {len(items)} v2 items accepted")
    print(f"  Verified: {passed_n} passed  |  {failed_n} dropped by verifier")
    print(f"  Flagged for priority QC (low confidence): {flagged}")
    print(f"  Model/gate verdict disagreements: {disagreed}  (see notes field)")
    print(f"\nOutput:     {output_path}")
    print(f"Audit log:  {log_path}")
    print(f"\nNext: uv run python scripts/qc_sample.py --pool v2 --n 20")


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
    parser.add_argument("--surplus-total", type=int, default=SURPLUS_TOTAL_DEFAULT,
                        help="Desired total candidate pool size: accepted_v1 + v2 candidates "
                             f"(default: {SURPLUS_TOTAL_DEFAULT})")
    parser.add_argument("--n-new", type=int, default=None,
                        help="Override: generate exactly this many new v2 candidates, "
                             "ignoring --surplus-total")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls; write placeholder items for testing")
    parser.add_argument(
        "--topics", type=str, default=None,
        help="Comma-separated topic CODES in PRIORITY order for a scoped top-up round "
             "(e.g. 'D3,D1,D5,C1,C3,C5'). First-listed wins ownership of a multi-tagged "
             "chunk. Omit to run all of TARGET_TOPICS in their normal order.",
    )
    parser.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH, dest="chunks_path",
                        help=f"Filtered chunks file to draw from (default: {CHUNKS_PATH})")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help=f"Where to write accepted items (default: {OUTPUT_PATH})")
    parser.add_argument("--log-path", type=Path, default=LOG_PATH, dest="log_path",
                        help=f"Where to write the audit log (default: {LOG_PATH})")
    args = parser.parse_args()

    topics = None
    if args.topics:
        codes = [c.strip().upper() for c in args.topics.split(",") if c.strip()]
        by_code = {t.split(".")[0]: t for t in TARGET_TOPICS}
        unknown = [c for c in codes if c not in by_code]
        if unknown:
            print(f"Unknown topic code(s): {unknown}. Valid codes: {sorted(by_code)}")
            sys.exit(1)
        topics = [by_code[c] for c in codes]

    main(args.model, args.verify_model, args.surplus_total, args.n_new, args.dry_run, args.seed,
         chunks_path=args.chunks_path, output_path=args.output, log_path=args.log_path, topics=topics)
