"""
Visualise the current English master selection.

Produces a 2x3 figure:
  Row 1: Source breakdown | ClimaQA topics | Climate-FEVER labels
  Row 2: CLINB topics     | CLINB IPCC level | Liu taxonomy coverage

Usage:
    uv run python scripts/plot_selection.py
    uv run python scripts/plot_selection.py --out figures/selection.png
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT           = Path(__file__).parent.parent
SELECTIONS_DIR = ROOT / "data" / "selections"

ALL_TOPICS = [
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

# Short labels for display
def short(t: str) -> str:
    return t.split(". ", 1)[1] if ". " in t else t


def load_items(fname: str) -> list[dict]:
    path = SELECTIONS_DIR / fname
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main(out_path: Path) -> None:
    climaqa = load_items("climaqa_selection.jsonl")
    fever   = load_items("climate_fever_selection.jsonl")
    pira    = load_items("pira_selection.jsonl")
    gen     = load_items("generated_mcq.jsonl")
    clinb   = load_items("clinb_selection.jsonl")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("English Master Selection — Current State", fontsize=14, fontweight="bold")

    # -----------------------------------------------------------------------
    # (0,0): overall breakdown by source
    # -----------------------------------------------------------------------
    ax = axes[0, 0]
    source_labels = []
    source_counts = []
    source_colors = []
    palette = {
        "climaqa_gold": "#4C72B0",
        "climate_fever": "#DD8452",
        "pira2":         "#55A868",
        "generated_mcq": "#C44E52",
        "clinb":         "#8172B2",
    }
    label_names = {
        "climaqa_gold":  f"ClimaQA-Gold\n({len(climaqa)} MCQ, 4-opt)",
        "climate_fever": f"Climate-FEVER\n({len(fever)} Claim)",
        "pira2":         f"PIRA 2.0\n({len(pira)} MCQ, 5-opt)",
        "generated_mcq": f"Generated MCQ\n({len(gen)})",
        "clinb":         f"CLINB\n({len(clinb)} Freeform)",
    }
    for src in ["climaqa_gold", "pira2", "generated_mcq", "climate_fever", "clinb"]:
        n = len([i for i in (climaqa + fever + pira + gen + clinb) if i.get("source") == src])
        if n > 0:
            source_labels.append(label_names[src])
            source_counts.append(n)
            source_colors.append(palette[src])

    wedges, texts, autotexts = ax.pie(
        source_counts,
        labels=source_labels,
        colors=source_colors,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
    total = sum(source_counts)
    ax.set_title(f"Items by Source  (total = {total})", fontweight="bold")

    # -----------------------------------------------------------------------
    # (0,1): ClimaQA topic distribution
    # -----------------------------------------------------------------------
    ax = axes[0, 1]
    topic_counts: Counter = Counter()
    untagged = 0
    for it in climaqa:
        topics = it.get("topic", [])
        if topics:
            for t in topics:
                topic_counts[t] += 1
        else:
            untagged += 1
    if untagged:
        topic_counts["(untagged)"] += untagged

    tc_labels = [short(t) for t in topic_counts]
    tc_values = list(topic_counts.values())

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(tc_labels), 1)) for i in range(len(tc_labels))]

    wedges, texts, autotexts = ax.pie(
        tc_values,
        labels=tc_labels,
        colors=colors,
        autopct=lambda p: f"{p:.0f}%" if p > 4 else "",
        startangle=90,
        pctdistance=0.8,
    )
    for t in texts:
        t.set_fontsize(7.5)
    for t in autotexts:
        t.set_fontsize(7.5)
    ax.set_title(f"ClimaQA Topics  ({len(climaqa)} items, counts = tag occurrences)", fontweight="bold")

    # -----------------------------------------------------------------------
    # (0,2): Climate-FEVER label distribution
    # -----------------------------------------------------------------------
    ax = axes[0, 2]
    fever_labels_map = {
        "SUPPORTS":        "#55A868",
        "REFUTES":         "#C44E52",
        "NOT_ENOUGH_INFO": "#8172B2",
        "DISPUTED":        "#CCB974",
    }
    fever_counts = Counter(it["gold"] for it in fever)
    fl = list(fever_counts.keys())
    fv = [fever_counts[k] for k in fl]
    fc = [fever_labels_map.get(k, "#aaa") for k in fl]

    wedges, texts, autotexts = ax.pie(
        fv,
        labels=[f"{k}\n({fever_counts[k]})" for k in fl],
        colors=fc,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title(f"Climate-FEVER Labels  ({len(fever)} items)", fontweight="bold")

    # -----------------------------------------------------------------------
    # (1,0): CLINB topic distribution
    # -----------------------------------------------------------------------
    ax = axes[1, 0]
    clinb_topic_counts: Counter = Counter(t for it in clinb for t in it.get("topic", []))
    clinb_topics  = list(clinb_topic_counts.keys())
    clinb_tvalues = [clinb_topic_counts[t] for t in clinb_topics]
    clinb_colors  = plt.get_cmap("Set2").colors[:len(clinb_topics)]

    wedges, texts, autotexts = ax.pie(
        clinb_tvalues,
        labels=[f"{t}\n({clinb_topic_counts[t]})" for t in clinb_topics],
        colors=clinb_colors,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title(f"CLINB Topics  ({len(clinb)} items)", fontweight="bold")

    # -----------------------------------------------------------------------
    # (1,1): CLINB IPCC confidence level distribution
    # -----------------------------------------------------------------------
    ax = axes[1, 1]
    level_map = {"High Confidence": "#55A868", "Advanced": "#4C72B0", "Open": "#CCB974"}
    level_counts: Counter = Counter(it.get("ipcc_confidence") for it in clinb)
    ll = list(level_counts.keys())
    lv = [level_counts[k] for k in ll]
    lc = [level_map.get(k, "#aaa") for k in ll]

    wedges, texts, autotexts = ax.pie(
        lv,
        labels=[f"{k}\n({level_counts[k]})" for k in ll],
        colors=lc,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("CLINB IPCC Evidence Level", fontweight="bold")

    # -----------------------------------------------------------------------
    # (1,2): Liu taxonomy coverage bar chart
    # -----------------------------------------------------------------------
    ax = axes[1, 2]

    covered_topics = {t for it in climaqa for t in it.get("topic", [])}
    # PIRA implicitly covers A3/B3 (ocean) -- mark those as "via PIRA"
    pira_topics = {"A3. Oceans, Cryosphere & Sea-Level Change",
                   "B3. Marine & Coastal Ecosystem Changes"}
    target_topics = {
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
    }

    bar_labels, bar_colors = [], []
    for t in ALL_TOPICS:
        bar_labels.append(t.split(". ")[0])   # just "A1", "B2", etc.
        if t in covered_topics:
            bar_colors.append("#4C72B0")   # blue = covered by ClimaQA
        elif t in pira_topics:
            bar_colors.append("#55A868")   # green = covered by PIRA
        elif t in target_topics:
            bar_colors.append("#C44E52")   # red = gap -- target for generation
        else:
            bar_colors.append("#CCB974")   # yellow = low-coverage, not primary target

    bar_values = [1] * len(ALL_TOPICS)
    bars = ax.barh(bar_labels[::-1], bar_values[::-1], color=bar_colors[::-1], height=0.7)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=8)

    legend_handles = [
        mpatches.Patch(color="#4C72B0", label="Covered by ClimaQA"),
        mpatches.Patch(color="#55A868", label="Covered by PIRA (ocean)"),
        mpatches.Patch(color="#C44E52", label="Gap — target for generation"),
        mpatches.Patch(color="#CCB974", label="Other (low priority)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5)
    ax.set_title("Liu Taxonomy Coverage (26 topics)", fontweight="bold")

    # -----------------------------------------------------------------------
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent.parent / "figures" / "selection_overview.png")
    args = parser.parse_args()
    main(args.out)
