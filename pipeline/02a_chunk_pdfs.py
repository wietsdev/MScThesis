"""
Stage 2a: Extract, chunk, and topic-tag IPCC AR6 Summary Volumes.

Reads:
    IPCC_AR6_WGII_SummaryVolume.pdf   (project root, or --wgii)
    IPCC_AR6_WGIII_SummaryVolume.pdf  (project root, or --wgiii)

Writes:
    data/sources/generation/chunks.jsonl
    Each line: {chunk_id, text, topics, source_pdf, page}

Only chunks that match at least one target Liu topic are written.
Topic tagging is keyword-based; review edge cases with --verbose.

Usage:
    uv run python pipeline/02a_chunk_pdfs.py
    uv run python pipeline/02a_chunk_pdfs.py --max-pages 300
    uv run python pipeline/02a_chunk_pdfs.py --wgii custom/path.pdf
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT     = Path(__file__).parent.parent
OUT_DIR  = ROOT / "data" / "sources" / "generation"
OUT_PATH = OUT_DIR / "chunks.jsonl"

CHUNK_WORDS = 450
OVERLAP     = 75
MIN_WORDS   = 150

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

# Keyword sets per target topic (lowercase, matched case-insensitively).
# A chunk is tagged to a topic if any of its keywords appears in the text.
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "B1. Biodiversity Loss": [
        "biodiversity", "species loss", "extinction", "habitat loss", "coral reef",
        "species richness", "endemic species", "range shift", "phenology",
        "ecosystem degradation",
    ],
    "C2. Water Resources & Hydrological Impacts": [
        "water resource", "groundwater", "water scarcity", "water stress",
        "hydrological", "water supply", "water security", "water availability",
        "aquifer", "streamflow", "drought impact", "water deficit",
    ],
    "C3. Human Health & Well-being": [
        "human health", "disease burden", "mortality", "heat stress", "air quality",
        "malaria", "dengue", "vector-borne", "mental health", "morbidity",
        "heat-related illness", "respiratory disease", "public health risk",
    ],
    "C4. Social Equity, Vulnerability & Migration": [
        "social equity", "vulnerability", "climate migration", "displacement",
        "climate justice", "inequality", "marginalised", "marginalized",
        "poverty", "indigenous peoples", "climate refugee", "gender equity",
        "social vulnerability", "adaptive capacity",
    ],
    "C5. Urban Systems & Infrastructure Impacts": [
        "urban system", "city infrastructure", "urban flood",
        "urban heat island", "critical infrastructure", "transport network",
        "power grid", "urban vulnerability", "built environment",
    ],
    "C6. Service & Industry Sector Impacts": [
        "tourism industry", "insurance sector", "financial risk",
        "supply chain disruption", "economic loss", "gdp impact",
        "economic sector", "service industry", "industry impact",
    ],
    "D1. Agricultural & Food System Adaptation": [
        "agricultural adaptation", "crop adaptation", "food system adaptation",
        "agroforestry", "drought-resistant", "heat-tolerant crop",
        "irrigation efficiency", "precision agriculture", "climate-smart agriculture",
        "adaptive capacity in agriculture", "crop switching",
    ],
    "D2. Urban Planning, Adaptation & Resilience": [
        "urban adaptation", "urban resilience", "green infrastructure",
        "flood-resilient", "nature-based solution", "adaptation planning",
        "urban regeneration", "urban climate resilience", "retrofitting cities",
    ],
    "D3. Public Health Adaptation": [
        "health adaptation", "disease surveillance", "heat action plan",
        "public health adaptation", "health system strengthening",
        "vector control", "climate-health adaptation", "early warning system for health",
    ],
    "D4. Public Awareness, Communication & Community Engagement": [
        "climate communication", "public awareness", "climate education",
        "community engagement", "citizen science", "climate literacy",
        "stakeholder engagement", "climate outreach", "public participation",
    ],
    "D5. Natural Resource Management & Conservation": [
        "natural resource management", "conservation", "protected area",
        "ecosystem-based adaptation", "sustainable land management",
        "forest management", "biodiversity conservation strategy",
        "sustainable fisheries",
    ],
    "E1. Climate Policy, Governance & Finance Mechanism": [
        "climate policy", "climate governance", "paris agreement",
        "unfccc", "nationally determined contribution", "ndc",
        "climate finance", "carbon tax", "carbon pricing",
        "adaptation finance", "climate legislation", "carbon market",
    ],
    "E2. Energy Transition": [
        "renewable energy", "solar power", "wind energy", "energy transition",
        "decarbonisation", "decarbonization", "net zero",
        "clean energy", "fossil fuel phase", "coal phase-out", "nuclear power",
        "hydrogen energy", "battery storage", "electricity grid",
        "energy system", "low-carbon energy",
    ],
    "E4. Land Use & Ecosystem-based Mitigation": [
        "land use change", "redd", "reforestation", "afforestation",
        "nature-based solution for mitigation", "soil carbon sequestration",
        "beccs", "blue carbon", "ecosystem-based mitigation",
        "land carbon sink", "forest carbon",
    ],
    "E5. Transport & Building Emissions Reduction": [
        "transport emissions", "electric vehicle", "ev adoption", "building efficiency",
        "aviation emissions", "shipping emissions", "low-carbon transport",
        "building retrofit", "sustainable mobility", "fuel efficiency standard",
        "zero-emission vehicle",
    ],
}


_FAQ_HEADER_RE = re.compile(
    r"^(FAQ\s+\d+\s+)?Frequently Asked Questions\s+(Frequently Asked Questions\s+)?"
    r"(FAQ\s+[\w.]+\s+\|\s+[^\n]+)?",
    re.IGNORECASE,
)


def clean_chunk(text: str) -> str:
    """Strip FAQ boilerplate headers that prepend some IPCC chunks."""
    return _FAQ_HEADER_RE.sub("", text).strip()


def tag_chunk(text: str) -> list[str]:
    """Return list of matching target topic names via keyword search."""
    text_lower = text.lower()
    return [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(kw in text_lower for kw in keywords)
    ]


def is_prose(text: str) -> bool:
    """
    Return True if the chunk looks like substantive prose.

    Rejects three chunk types that produce poor MCQ:
    1. Tables / bullet lists: few sentence-ending punctuation marks,
       short avg words-per-sentence.
    2. Figure / table descriptions: text describing axes, legends, or
       confidence-level keys rather than factual content.
    3. Glossary / annex sections: definition blocks that produce trivial
       "what is the definition of X" questions.
    """
    sentence_ends = sum(1 for c in text if c in ".?!")
    if sentence_ends < 3:
        return False
    words = len(text.split())
    if words / sentence_ends < 8:
        return False

    text_lower = text.lower()

    # Reject figure / table description chunks
    figure_markers = [
        "figure ts.", "figure box", "table ts.", "figure spm.",
        "confidence level", "likelihood level",
        "| synthesis of", "legend:", "note:", "source: ipcc",
    ]
    if any(m in text_lower for m in figure_markers):
        return False

    # Reject glossary / annex blocks (definitions)
    glossary_markers = [
        "annex i", "annex ii", "glossary", "see also ",
        "also known as", "(ipcc ", "(wmo ", "(unep ",
    ]
    glossary_hits = sum(1 for m in glossary_markers if m in text_lower)
    if glossary_hits >= 2:
        return False

    return True


def extract_and_chunk(
    pdf_path: Path,
    source_label: str,
    max_pages: int | None,
    verbose: bool,
) -> list[dict]:
    try:
        import fitz
    except ImportError:
        print("ERROR: pymupdf not installed. Run: uv add pymupdf")
        sys.exit(1)

    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    limit = min(n_pages, max_pages) if max_pages else n_pages
    print(f"  {pdf_path.name}: {n_pages} pages, processing {limit}")

    chunks: list[dict] = []
    chunk_idx = 0

    for page_num in range(limit):
        page_text = doc[page_num].get_text("text")
        words = page_text.split()
        if len(words) < MIN_WORDS:
            continue

        stride = CHUNK_WORDS - OVERLAP
        i = 0
        while i < len(words):
            w_slice = words[i : i + CHUNK_WORDS]
            if len(w_slice) < MIN_WORDS:
                break
            chunk_text = clean_chunk(" ".join(w_slice))
            if len(chunk_text.split()) < MIN_WORDS:
                i += stride
                continue
            topics = tag_chunk(chunk_text)
            if topics and is_prose(chunk_text):
                chunk_id = f"{source_label}_p{page_num + 1:04d}_c{chunk_idx:04d}"
                chunks.append({
                    "chunk_id":   chunk_id,
                    "text":       chunk_text,
                    "topics":     topics,
                    "source_pdf": source_label,
                    "page":       page_num + 1,
                })
                if verbose:
                    print(f"    {chunk_id}  topics={topics}")
                chunk_idx += 1
            i += stride

    doc.close()
    return chunks


def main(
    wgii_path: Path,
    wgiii_path: Path,
    max_pages: int | None,
    verbose: bool,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []

    for path, label in [(wgii_path, "wgii"), (wgiii_path, "wgiii")]:
        if path.exists():
            chunks = extract_and_chunk(path, label, max_pages, verbose)
            print(f"  -> {len(chunks)} tagged chunks from {label.upper()}")
            all_chunks.extend(chunks)
        else:
            print(f"  {label.upper()} PDF not found: {path} -- skipping")

    if not all_chunks:
        print("No chunks extracted. Check PDF paths.")
        sys.exit(1)

    OUT_PATH.write_text("\n".join(json.dumps(c) for c in all_chunks) + "\n")

    topic_counts = Counter(t for c in all_chunks for t in c["topics"])
    print(f"\nTotal chunks written: {len(all_chunks)}")
    print("\nChunks per target topic:")
    for topic in TARGET_TOPICS:
        n = topic_counts.get(topic, 0)
        bar = "#" * min(n // 10, 40)
        print(f"  {topic[:48]:<48}  {n:4d}  {bar}")

    print(f"\nOutput: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wgii",  type=Path,
        default=ROOT / "IPCC_AR6_WGII_SummaryVolume.pdf",
    )
    parser.add_argument(
        "--wgiii", type=Path,
        default=ROOT / "IPCC_AR6_WGIII_SummaryVolume.pdf",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Max pages to read per PDF (default: all)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args.wgii, args.wgiii, args.max_pages, args.verbose)
