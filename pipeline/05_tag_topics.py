"""
Stage 5: Backfill liu_topic across every selection so topic-diversity analysis
can span the whole benchmark, not just the three sources that already carry
Liu et al. taxonomy tags.

Liu et al.'s own tagging method (arXiv:2603.26106) is itself an LLM few-shot
classification pipeline against a fixed taxonomy (GPT-4.1-mini for free-form
generation, GPT-5-mini with 4-shot in-context learning for reassignment to
the final 26 topics, up to 3 labels per item) -- not a deterministic rule set.
There is nothing to "exactly reproduce" beyond running a similarly-structured
classification prompt against the same fixed taxonomy; this script does that
with a gateway model instead of GPT-4.1-mini/GPT-5-mini specifically.

Two different things happen per source:
  - ClimaQA / Climate-FEVER / generated MCQ already carry Liu tags in `topic`
    -- liu_topic is just copied across, no LLM call.
  - PIRA (topic=[], untagged by design) and CLINB (topic=its own six-category
    system, NOT Liu) get a fresh LLM classification call each.

Reads / writes IN PLACE:
    data/selections/climaqa_selection.jsonl
    data/selections/climate_fever_selection.jsonl
    data/selections/pira_selection.jsonl
    data/selections/clinb_selection.jsonl
    data/selections/accepted_v1.jsonl
    data/selections/accepted_v2.jsonl

Usage:
    uv run python pipeline/05_tag_topics.py
    uv run python pipeline/05_tag_topics.py --dry-run --n 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from item_schema import Item

ROOT           = Path(__file__).parent.parent
SELECTIONS_DIR = ROOT / "data" / "selections"

TAG_MODEL = "claude-sonnet"   # non-SUT, already used for the verifier role

ALREADY_TAGGED_FILES = [
    "climaqa_selection.jsonl",
    "climate_fever_selection.jsonl",
    "accepted_v1.jsonl",
    "accepted_v2.jsonl",
]
UNTAGGED_FILES = [
    "pira_selection.jsonl",
    "clinb_selection.jsonl",
]

# One-line descriptions written for this project -- not verbatim from the
# paper (Appendix A's exact wording isn't published), but the same 26 fixed
# topic codes/names already used throughout this codebase.
TAXONOMY = {
    "A1. Atmospheric Science & Climate Processes": "physical climate system dynamics, atmospheric composition, radiative forcing",
    "A2. Greenhouse Gas & Biogeochemical Cycles": "GHG sources and sinks, carbon/nitrogen cycles, emissions accounting",
    "A3. Oceans, Cryosphere & Sea-Level Change": "ocean warming/acidification, ice sheets and glaciers, sea level rise",
    "A4. Extreme Weather Events": "heatwaves, storms, droughts, floods and their attribution or trends",
    "A5. Climate Modeling": "climate models, emissions scenarios (SSPs/RCPs), projections and their uncertainty",
    "A6. Environmental Monitoring": "observation systems, satellite/sensor data, detection of environmental change",
    "B1. Biodiversity Loss": "species extinction, habitat loss, ecosystem degradation from climate change",
    "B2. Terrestrial & Freshwater Ecosystem Changes": "forests, rivers, lakes and other land ecosystems responding to climate change",
    "B3. Marine & Coastal Ecosystem Changes": "coral reefs, fisheries, coastal and ocean ecosystems responding to climate change",
    "C1. Agriculture & Food Security": "crop yields, food supply, farming impacted by climate change (impacts, not adaptation actions)",
    "C2. Water Resources & Hydrological Impacts": "water scarcity, supply, quality, hydrological cycle impacts",
    "C3. Human Health & Well-being": "disease, mortality, mental health impacts of climate change",
    "C4. Social Equity, Vulnerability & Migration": "inequality, displacement, vulnerable populations, climate justice",
    "C5. Urban Systems & Infrastructure Impacts": "cities, buildings, transport/power infrastructure impacted by climate change",
    "C6. Service & Industry Sector Impacts": "tourism, insurance, and other economic sector impacts",
    "D1. Agricultural & Food System Adaptation": "adaptation actions and strategies in farming and food systems",
    "D2. Urban Planning, Adaptation & Resilience": "city-level adaptation, resilient infrastructure, urban planning responses",
    "D3. Public Health Adaptation": "health-system adaptation actions, disease surveillance, heat action plans",
    "D4. Public Awareness, Communication & Community Engagement": "climate education, communication, community-level engagement",
    "D5. Natural Resource Management & Conservation": "conservation strategies, protected areas, ecosystem-based adaptation",
    "E1. Climate Policy, Governance & Finance Mechanism": "international agreements, national policy, climate finance mechanisms",
    "E2. Energy Transition": "renewable energy, decarbonising the energy system, energy technology",
    "E3. Corporate & Industry Climate Action": "company or sector-level mitigation commitments, greenwashing, industry decarbonisation",
    "E4. Land Use & Ecosystem-based Mitigation": "reforestation, land-use change, carbon removal via land or ecosystems",
    "E5. Transport & Building Emissions Reduction": "decarbonising transport and buildings specifically",
    "F1. Others": "doesn't fit any topic above",
}

TAG_PROMPT = """\
You are tagging a climate-related item with topics from a fixed taxonomy, in \
the style of a benchmark topic taxonomy for climate NLP datasets.

Assign 1 to 3 topics from the list below that best describe what this item is \
fundamentally about. Rank them most-relevant first. Only use topics from this \
exact list -- do not invent new ones.

TAXONOMY:
{taxonomy_block}

ITEM TEXT:
"{text}"

Return ONLY the code and name for each topic (e.g. "A5. Climate Modeling"), \
never the description text after the "--".

Respond with valid JSON only (no markdown fences):
{{"topics": ["<code and name only, e.g. 'A5. Climate Modeling'>", "..."]}}
"""


def _taxonomy_block() -> str:
    return "\n".join(f"{code} -- {desc}" for code, desc in TAXONOMY.items())


def _parse_json(raw: str) -> dict | None:
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def tag_item(text: str, dry_run: bool) -> list[str]:
    if dry_run:
        return ["F1. Others"]

    import models
    prompt = TAG_PROMPT.format(taxonomy_block=_taxonomy_block(), text=text[:1500])
    raw = models.generate(prompt, TAG_MODEL, max_tokens=150, temperature=0.0)
    result = _parse_json(raw)
    if result is None or "topics" not in result:
        return []

    def normalize(t: str) -> str | None:
        if t in TAXONOMY:
            return t
        # model sometimes echoes "CODE. Name -- description" despite the
        # instruction not to -- recover the code+name prefix before matching
        prefix = t.split(" -- ")[0].strip()
        return prefix if prefix in TAXONOMY else None

    valid = [normalize(t) for t in result["topics"]]
    valid = [t for t in valid if t is not None]
    return valid[:3]


def process_already_tagged(fname: str) -> tuple[int, int]:
    path = SELECTIONS_DIR / fname
    if not path.exists():
        return 0, 0
    items = [Item.from_dict(json.loads(l)) for l in path.read_text().splitlines() if l.strip()]
    for it in items:
        it.liu_topic = list(it.topic)
    path.write_text("\n".join(json.dumps(it.to_dict()) for it in items) + "\n")
    return len(items), sum(1 for it in items if it.liu_topic)


def process_untagged(fname: str, n: int | None, dry_run: bool) -> tuple[int, int]:
    path = SELECTIONS_DIR / fname
    if not path.exists():
        return 0, 0
    items = [Item.from_dict(json.loads(l)) for l in path.read_text().splitlines() if l.strip()]
    target = items[:n] if n is not None else items

    tagged = 0
    for i, it in enumerate(target, 1):
        it.liu_topic = tag_item(it.question, dry_run)
        if it.liu_topic:
            tagged += 1
        if i % 25 == 0 or i == len(target):
            print(f"    {i}/{len(target)}  ({tagged} tagged so far)")

    path.write_text("\n".join(json.dumps(it.to_dict()) for it in items) + "\n")
    return len(target), tagged


def main(n: int | None, dry_run: bool) -> None:
    print("Backfilling liu_topic (copy, no LLM call):")
    for fname in ALREADY_TAGGED_FILES:
        total, tagged = process_already_tagged(fname)
        if total:
            print(f"  {fname}: {tagged}/{total} items now have liu_topic")
        else:
            print(f"  {fname}: not found, skipping")

    print(f"\nTagging untagged sources with {TAG_MODEL} (LLM classification):")
    for fname in UNTAGGED_FILES:
        print(f"\n  {fname}")
        total, tagged = process_untagged(fname, n, dry_run)
        if total:
            print(f"  -> {tagged}/{total} items tagged")
        else:
            print(f"  -> not found, skipping")

    print("\nDone. Re-run pipeline/03_freeze.py to fold this into a new master version.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Only tag the first N items per untagged file (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls; tag everything F1. Others (for testing)")
    args = parser.parse_args()
    main(args.n, args.dry_run)
