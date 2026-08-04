"""
Visualise generated MCQ topic distribution against the existing benchmark.

Produces figures/generation_overview.png:
  Left:  horizontal bar — generated items per Liu topic
  Right: stacked horizontal bar — all 26 topics (existing blue | generated orange)

Usage:
    MPLBACKEND=Agg uv run python scripts/plot_generation.py
    MPLBACKEND=Agg uv run python scripts/plot_generation.py --out figures/generation.png
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT           = Path(__file__).parent.parent
SELECTIONS_DIR = ROOT / "data" / "selections"

ALL_26_TOPICS = [
    "A1. Atmospheric Science & Climate Processes",
    "A2. Greenhouse Gas & Biogeochemical Cycles",
    "A3. Oceans, Cryosphere & Sea-Level Change",
    "A4. Extreme Weather Events",
    "A5. Climate Modeling",
    "A6. Environmental Monitoring",
    "B1. Biodiversity Loss",
    "B2. Terrestrial & Freshwater Ecosystem Changes",
    "B3. Marine & Coastal Ecosystem Changes",
    "C1. Agriculture & Food Security",
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
    "E3. Corporate & Industry Climate Action",
    "E4. Land Use & Ecosystem-based Mitigation",
    "E5. Transport & Building Emissions Reduction",
    "F1. Others",
]

EXISTING_SELECTIONS = [
    "climaqa_selection.jsonl",
    "climate_fever_selection.jsonl",
    "pira_selection.jsonl",
]


def load_items(fname: str) -> list[dict]:
    path = SELECTIONS_DIR / fname
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def short(t: str) -> str:
    return t.split(". ", 1)[0]  # just "A1", "B2", etc.


def main(out_path: Path) -> None:
    existing_items = []
    for fname in EXISTING_SELECTIONS:
        existing_items.extend(load_items(fname))

    generated_items = load_items("generated_mcq.jsonl")

    existing_counts  = Counter(t for it in existing_items  for t in it.get("topic", []))
    generated_counts = Counter(t for it in generated_items for t in it.get("topic", []))

    target_topics = [t for t in ALL_26_TOPICS if generated_counts.get(t, 0) > 0]

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle(
        f"Generated MCQ: {len(generated_items)} new items  "
        f"(total benchmark: {len(existing_items) + len(generated_items)})",
        fontsize=13, fontweight="bold",
    )

    # ---- Left: generated items per topic -----------------------------------
    ax = axes[0]
    gen_vals    = [generated_counts.get(t, 0) for t in target_topics]
    gen_labels  = [t.split(". ", 1)[1] if ". " in t else t for t in target_topics]
    sorted_idx  = sorted(range(len(gen_vals)), key=lambda i: gen_vals[i])
    s_labels    = [gen_labels[i] for i in sorted_idx]
    s_vals      = [gen_vals[i] for i in sorted_idx]

    bars = ax.barh(s_labels, s_vals, color="#DD8452", edgecolor="white", height=0.7)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Items generated", fontsize=11)
    ax.set_title("Generated MCQ by Liu Topic", fontweight="bold")
    ax.set_xlim(0, max(s_vals) * 1.25 if s_vals else 5)

    # ---- Right: stacked bar across all 26 topics ---------------------------
    ax = axes[1]
    topic_codes = [short(t) for t in ALL_26_TOPICS]
    ex_vals  = [existing_counts.get(t, 0)  for t in ALL_26_TOPICS]
    gen_vals2 = [generated_counts.get(t, 0) for t in ALL_26_TOPICS]

    # Reverse order so A1 is at top
    rev_labels  = topic_codes[::-1]
    rev_ex      = ex_vals[::-1]
    rev_gen     = gen_vals2[::-1]

    ax.barh(rev_labels, rev_ex,  color="#4C72B0", height=0.7, label="Existing (ClimaQA + FEVER + PIRA)")
    ax.barh(rev_labels, rev_gen, color="#DD8452", height=0.7,
            left=rev_ex, label=f"Generated ({len(generated_items)})")

    ax.set_xlabel("Items (topic tag occurrences)", fontsize=11)
    ax.set_title("Overall Benchmark Coverage — All 26 Liu Topics", fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)

    # Vertical line at 8 (thin coverage threshold)
    ax.axvline(x=8, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="target ≥ 8")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Console summary
    print(f"\n{'Topic':<48}  existing  generated  total")
    print("─" * 70)
    for t in ALL_26_TOPICS:
        ex  = existing_counts.get(t, 0)
        gen = generated_counts.get(t, 0)
        if ex + gen == 0:
            continue
        tag = "  ← THIN" if ex + gen < 8 else ""
        print(f"  {t:<48}  {ex:6d}  {gen:9d}  {ex+gen:5d}{tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "figures" / "generation_overview.png",
    )
    args = parser.parse_args()
    main(args.out)
