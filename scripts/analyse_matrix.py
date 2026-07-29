"""
Analyse a matrix run: per-model accuracy, MCQ complexity breakdown,
verbose-response examples, and a cross-model comparison table.

Usage:
    uv run python scripts/analyse_matrix.py results/matrix_<ts>
    uv run python scripts/analyse_matrix.py          # uses most recent matrix
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
MCQ_VERBOSE_CHARS = 10   # MCQ response longer than this is flagged as verbose
CLOZE_VERBOSE_CHARS = 50


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_latest_matrix() -> Path:
    candidates = sorted(RESULTS_DIR.glob("matrix_*"))
    if not candidates:
        raise FileNotFoundError(f"No matrix runs found in {RESULTS_DIR}")
    return candidates[-1]


def load_matrix_config(matrix_dir: Path) -> dict:
    return json.loads((matrix_dir / "matrix_config.json").read_text())


def load_scores(matrix_dir: Path, model_id: str, split: str) -> list[dict]:
    path = matrix_dir / f"{model_id}_{split}" / "scores.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tally(scores: list[dict]) -> dict:
    n_correct     = sum(1 for s in scores if s["correct"] is True)
    n_wrong       = sum(1 for s in scores if s["correct"] is False)
    n_unparseable = sum(1 for s in scores if s["correct"] == "unparseable")
    n_total       = len(scores)
    accuracy      = n_correct / n_total if n_total else 0.0
    return {
        "total": n_total,
        "correct": n_correct,
        "wrong": n_wrong,
        "unparseable": n_unparseable,
        "accuracy": accuracy,
    }


def accuracy_by_field(scores: list[dict], field: str) -> dict:
    """
    Group scores by a field value and compute a tally per group.
    Works for any field present in the scored records (e.g. complexity, topic).
    Returns {field_value: tally_dict}, sorted by field value.
    """
    groups = defaultdict(list)
    for s in scores:
        key = s.get(field) or "unknown"
        groups[key].append(s)
    return {k: tally(groups[k]) for k in sorted(groups)}


# ---------------------------------------------------------------------------
# Verbose-response detection
# ---------------------------------------------------------------------------

def verbose_examples(scores: list[dict], split: str, n: int = 2) -> list[dict]:
    """Return up to n scored records where the raw response looks verbose."""
    limit = MCQ_VERBOSE_CHARS if split == "mcq" else CLOZE_VERBOSE_CHARS
    return [
        s for s in scores if len(s.get("raw_response", "")) > limit
    ][:n]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def pct(t: dict) -> str:
    return f"{t['accuracy']:.0%}"


def print_tally(t: dict) -> None:
    print(
        f"    Accuracy: {t['accuracy']:.1%}  "
        f"({t['correct']} correct / {t['wrong']} wrong / {t['unparseable']} unparseable)"
    )


def print_breakdown(breakdown: dict, field: str) -> None:
    print(f"    By {field}:")
    for value, t in breakdown.items():
        bar = t["correct"] * "+" + t["wrong"] * "-" + t["unparseable"] * "?"
        print(f"      {value:<15} {t['accuracy']:.0%}  {bar}  (n={t['total']})")


def print_verbose(examples: list[dict], split: str) -> None:
    if not examples:
        return
    print(f"    Verbose responses (>{MCQ_VERBOSE_CHARS if split == 'mcq' else CLOZE_VERBOSE_CHARS} chars):")
    for e in examples:
        print(f"      raw: {e['raw_response']!r}")
        print(f"      parsed: {e['parsed']!r}  gold: {e['gold']!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(matrix_dir: Path) -> None:
    cfg = load_matrix_config(matrix_dir)
    models = cfg["models"]
    splits = cfg["splits"]

    print(f"Matrix run: {matrix_dir.name}")
    print(f"Models: {models}")
    print(f"Items:  {cfg['n_items']} per split  (seed={cfg['seed']})")

    # --- per-model detail ---
    all_tallies = {}   # (model_id, split) -> tally dict, for the comparison table

    for model_id in models:
        print(f"\n{'=' * 55}")
        print(f"  {model_id}")
        print(f"{'=' * 55}")
        for split in splits:
            scores = load_scores(matrix_dir, model_id, split)
            t = tally(scores)
            all_tallies[(model_id, split)] = t

            print(f"\n  {split.upper()}")
            print_tally(t)

            if split == "mcq":
                breakdown = accuracy_by_field(scores, "complexity")
                print_breakdown(breakdown, "complexity")

            examples = verbose_examples(scores, split)
            print_verbose(examples, split)

    # --- comparison table ---
    print(f"\n{'=' * 55}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 55}")

    col = 14
    header = f"  {'Model':<20}"
    for split in splits:
        header += f"  {split.upper() + ' acc':>{col}}  {'unparse':>7}"
    print(header)
    print("  " + "-" * (20 + len(splits) * (col + 11)))

    for model_id in models:
        row = f"  {model_id:<20}"
        for split in splits:
            t = all_tallies[(model_id, split)]
            row += f"  {pct(t):>{col}}  {t['unparseable']:>7}"
        print(row)

    print()


def compare(dir_a: Path, dir_b: Path) -> None:
    """Print a side-by-side accuracy table for two matrix runs."""
    cfg_a = load_matrix_config(dir_a)
    cfg_b = load_matrix_config(dir_b)
    models = cfg_a["models"]
    splits = cfg_a["splits"]

    label_a = f"{dir_a.name} (n={cfg_a['n_items']})"
    label_b = f"{dir_b.name} (n={cfg_b['n_items']})"

    print(f"\n{'=' * 70}")
    print("  CROSS-RUN COMPARISON")
    print(f"  A: {label_a}")
    print(f"  B: {label_b}")
    print(f"{'=' * 70}")

    col = 10
    header = f"  {'Model':<20}"
    for split in splits:
        header += f"  {split.upper() + ' A':>{col}}  {split.upper() + ' B':>{col}}  {'diff':>6}"
    print(header)
    print("  " + "-" * (20 + len(splits) * (col * 2 + 16)))

    for model_id in models:
        row = f"  {model_id:<20}"
        for split in splits:
            try:
                scores_a = load_scores(dir_a, model_id, split)
                scores_b = load_scores(dir_b, model_id, split)
                t_a = tally(scores_a)
                t_b = tally(scores_b)
                diff = t_b["accuracy"] - t_a["accuracy"]
                sign = "+" if diff >= 0 else ""
                row += f"  {pct(t_a):>{col}}  {pct(t_b):>{col}}  {sign}{diff:.0%}".replace("+0%", " 0%")
            except FileNotFoundError:
                row += f"  {'n/a':>{col}}  {'n/a':>{col}}  {'':>6}"
        print(row)
    print()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        dir_a = Path(sys.argv[1])
        dir_b = Path(sys.argv[2])
        for d in (dir_a, dir_b):
            if not d.exists():
                print(f"Error: {d} does not exist.")
                sys.exit(1)
        compare(dir_a, dir_b)
    elif len(sys.argv) == 2:
        matrix_dir = Path(sys.argv[1])
        if not matrix_dir.exists():
            print(f"Error: {matrix_dir} does not exist.")
            sys.exit(1)
        main(matrix_dir)
    else:
        matrix_dir = find_latest_matrix()
        print(f"(No path given -- using latest: {matrix_dir.name})\n")
        main(matrix_dir)
