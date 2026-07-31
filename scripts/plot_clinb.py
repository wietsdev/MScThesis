"""
Reproduce CLINB paper Figure 4: topic x difficulty level, split by IPCC Working Group.

Usage:
    uv run python scripts/plot_clinb.py
    uv run python scripts/plot_clinb.py --out figures/clinb_overview.png
"""

import argparse
import ast
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent


def load_df() -> pd.DataFrame:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "deepmind/clinb-questions",
        "CLINB-questions.csv",
    )
    df["level"] = df["level"].apply(ast.literal_eval)
    df["wg"]    = df["wg"].apply(ast.literal_eval)
    df["topic"] = df["topic"].apply(ast.literal_eval)
    return df


def main(out_path: Path) -> None:
    df = load_df()
    exploded = df.explode("level").explode("topic").explode("wg")

    topics = sorted(exploded["topic"].unique())
    levels = sorted(exploded["level"].unique())

    level_colors = {
        "Advanced":        "tab:blue",
        "High Confidence": "tab:orange",
        "Open":            "tab:green",
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
    fig.suptitle(
        "Distribution of Topics and Difficulty Levels by Working Group",
        y=1.03,
        fontsize=28,
    )

    for i, wg in enumerate(["I", "II", "III"]):
        wg_df = exploded[exploded["wg"] == wg]
        counts = (
            wg_df.groupby(["topic", "level"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=topics, columns=levels, fill_value=0)
        )
        counts.plot(
            kind="bar",
            stacked=True,
            ax=axes[i],
            color=[level_colors[lv] for lv in levels],
            width=0.8,
            edgecolor="black",
            linewidth=1.5,
        )
        axes[i].set_title(f"Working Group: {wg}", fontsize=26)
        axes[i].set_xlabel("Topic", fontsize=22)
        axes[i].xaxis.grid(False)
        axes[i].tick_params(axis="x", rotation=80, labelsize=20)
        axes[i].tick_params(axis="y", labelsize=20)

        if i == 0:
            axes[i].set_ylabel("Number of Questions", fontsize=24)
        else:
            axes[i].set_ylabel("")

        if i == 1:
            axes[i].legend(title="Level", loc="upper right", fontsize=18, title_fontsize=20)
        else:
            axes[i].get_legend().remove()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "figures" / "clinb_overview.png")
    args = parser.parse_args()
    main(args.out)
