"""
Stage 2: Generate 100 new MCQ items from source documents.

Purpose: fill topic gaps not covered by ClimaQA-Gold or PIRA 2.0.
Generated items must NOT originate from the same RAG index used for
evaluation (contamination constraint).

Intended source documents (download separately before running):
    - IPCC AR6 Working Group II (Adaptation) report
    - IPCC AR6 Working Group III (Mitigation) report
    - Skeptical Science rebuttal database
    - ClimateQ&A Conversational Corpus
    - Carbon Disclosure Project (CLIMA-CDP) reports

Output:
    data/selections/generated_mcq.jsonl   (100 MCQ, 4-option)

Usage:
    uv run python pipeline/02_generate_mcq.py --source-dir data/sources/generation

Status: STUB -- interface defined, generation not yet implemented.
        Run pipeline/01_select.py first to see which topics need filling.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from item_schema import Item

ROOT           = Path(__file__).parent.parent
SELECTIONS_DIR = ROOT / "data" / "selections"
SOURCES_DIR    = ROOT / "data" / "sources" / "generation"
OUTPUT_PATH    = SELECTIONS_DIR / "generated_mcq.jsonl"

N_GENERATE = 100
SEED       = 42

# Topics absent from ClimaQA MCQ -- generation must cover these.
# Derived from 01_select.py coverage report (ClimaQA only covers A1-A6, B2-B3, C1, E3, F1).
TARGET_TOPICS: list[str] = [
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

GENERATION_PROMPT = """\
You are a climate science expert creating multiple-choice exam questions.

Using ONLY the passage below, write one multiple-choice question with exactly 4 options \
(a, b, c, d). The question should test understanding of a specific fact or concept in the \
passage. The correct answer should be unambiguous.

Format your response as JSON with these fields:
  question   - the question text
  a, b, c, d - the four option texts
  gold       - the correct letter (a/b/c/d)
  excerpt    - the verbatim sentence(s) from the passage that support the answer

Passage:
{passage}
"""


def generate_from_passage(passage: str, model_id: str, source_doc_id: str) -> Item | None:
    """Generate one MCQ from a passage using the gateway LLM. Returns None on failure."""
    import models

    prompt = GENERATION_PROMPT.format(passage=passage)
    raw = models.generate(prompt, model_id, max_tokens=512, temperature=0.3)

    try:
        # Strip markdown code fences if present
        text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    required = {"question", "a", "b", "c", "d", "gold"}
    if not required.issubset(data):
        return None

    gold = data["gold"].strip().lower()
    if gold not in {"a", "b", "c", "d"}:
        return None

    return Item(
        item_id       = f"generated_mcq_{source_doc_id}",
        source        = "generated_mcq",
        license       = "generated",   # update with source doc license before publishing
        item_type     = "mcq",
        language      = "en",
        question      = data["question"],
        gold          = gold,
        options       = {k: data[k] for k in ("a", "b", "c", "d")},
        source_doc_id = source_doc_id,
        source_excerpt = data.get("excerpt"),
    )


def load_passages(source_dir: Path) -> list[tuple[str, str]]:
    """
    Load (passage_text, doc_id) pairs from source_dir.
    Expected layout: one .txt file per chunk, filename used as doc_id.
    Populate data/sources/generation/ before running.
    """
    passages = []
    for txt_file in sorted(source_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        if text:
            passages.append((text, txt_file.stem))
    return passages


def main(source_dir: Path, model_id: str) -> None:
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        print("Create it and populate with .txt passage files before running.")
        print("See module docstring for intended source documents.")
        sys.exit(1)

    passages = load_passages(source_dir)
    if not passages:
        print(f"No .txt files found in {source_dir}")
        sys.exit(1)

    print(f"Loaded {len(passages)} passages from {source_dir}")

    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    items: list[Item] = []
    failed = 0
    for i, (passage, doc_id) in enumerate(passages):
        if len(items) >= N_GENERATE:
            break
        print(f"  [{i+1}/{len(passages)}] {doc_id}  ", end="", flush=True)
        item = generate_from_passage(passage, model_id, f"{doc_id}_{i:04d}")
        if item:
            items.append(item)
            print("ok")
        else:
            failed += 1
            print("failed (parse error)")

    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(it.to_dict()) for it in items) + "\n"
    )
    print(f"\nGenerated: {len(items)}  Failed: {failed}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCES_DIR)
    parser.add_argument("--model", default="claude-haiku",
                        help="Gateway model ID to use for generation")
    args = parser.parse_args()
    main(args.source_dir, args.model)
