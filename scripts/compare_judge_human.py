"""
Human-vs-LLM-judge agreement on the N=27 English CLINB judge QC sample.

Workflow (blind-first, to avoid anchoring on the judge's own scores):
  1. Fill in judge_qc_sample_blind.csv's your_* columns (no judge_* columns
     visible in that file -- that's the point).
  2. Run this script. It merges your scores back against the judge's actual
     verdicts (held in the original judge_qc_sample.csv, sitting alongside
     the blind file) and reports per-dimension agreement.

Metrics reported per dimension (accuracy, calibration, completeness,
groundedness, harm_avoidance, overall), each on a 1-5 scale:
  - exact agreement rate, within-1 agreement rate
  - mean absolute difference (human - judge)
  - Pearson r
  - quadratic weighted Cohen's kappa (chance-corrected agreement for ordinal
    scales -- penalizes a 1-vs-5 disagreement far more than a 3-vs-4, unlike
    plain kappa)

No scipy/sklearn dependency -- consistent with the rest of this project's
hand-implemented stats (see plot_composite_score_portrait.py's spearman()).

Usage:
    uv run python scripts/compare_judge_human.py \\
        --run-dir results/20260820-144026_gemma3-12b_full
"""

import argparse
import csv
from pathlib import Path

DIMENSIONS = ["accuracy", "calibration", "completeness", "groundedness", "harm_avoidance", "overall"]


def load_csv(p: Path) -> dict[str, dict]:
    with open(p, newline="") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def to_int(v: str):
    v = (v or "").strip()
    return int(v) if v else None


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy) ** 0.5


def quadratic_weighted_kappa(xs: list[int], ys: list[int], min_r=1, max_r=5) -> float:
    n_cats = max_r - min_r + 1
    hist_x = [0] * n_cats
    hist_y = [0] * n_cats
    observed = [[0] * n_cats for _ in range(n_cats)]
    for x, y in zip(xs, ys):
        hist_x[x - min_r] += 1
        hist_y[y - min_r] += 1
        observed[x - min_r][y - min_r] += 1
    n = len(xs)
    weights = [[((i - j) ** 2) / ((n_cats - 1) ** 2) for j in range(n_cats)] for i in range(n_cats)]
    expected = [[hist_x[i] * hist_y[j] / n for j in range(n_cats)] for i in range(n_cats)]
    num = sum(weights[i][j] * observed[i][j] for i in range(n_cats) for j in range(n_cats))
    den = sum(weights[i][j] * expected[i][j] for i in range(n_cats) for j in range(n_cats))
    if den == 0:
        return float("nan")
    return 1 - num / den


def main(run_dir: Path) -> None:
    judge_rows = load_csv(run_dir / "judge_qc_sample.csv")
    blind_rows = load_csv(run_dir / "judge_qc_sample_blind.csv")

    missing = [iid for iid, r in blind_rows.items() if not (r.get("your_overall") or "").strip()]
    if missing:
        print(f"[WARN] {len(missing)}/{len(blind_rows)} items still unscored in judge_qc_sample_blind.csv: {missing}")
        print("Fill those in before trusting these numbers -- partial results below use only the scored subset.\n")

    print(f"{'dimension':<15} {'n':>3} {'exact':>7} {'within1':>8} {'mean diff':>10} {'pearson r':>10} {'qw kappa':>9}")
    for dim in DIMENSIONS:
        pairs = []
        for iid, jr in judge_rows.items():
            br = blind_rows.get(iid)
            if br is None:
                continue
            j = to_int(jr[f"judge_{dim}"])
            h = to_int(br[f"your_{dim}"])
            if j is None or h is None:
                continue
            pairs.append((j, h))
        if not pairs:
            print(f"{dim:<15} {'--':>3}   (no scored items yet)")
            continue
        js = [p[0] for p in pairs]
        hs = [p[1] for p in pairs]
        n = len(pairs)
        exact = sum(1 for j, h in pairs if j == h) / n
        within1 = sum(1 for j, h in pairs if abs(j - h) <= 1) / n
        mean_diff = sum(h - j for j, h in pairs) / n
        r = pearson([float(x) for x in js], [float(x) for x in hs]) if n > 1 else float("nan")
        kappa = quadratic_weighted_kappa(js, hs) if n > 1 else float("nan")
        print(f"{dim:<15} {n:>3} {exact:>7.0%} {within1:>8.0%} {mean_diff:>+10.2f} {r:>10.2f} {kappa:>9.2f}")

    print("\nmean diff = your_score - judge_score (positive = you scored higher than the judge)")
    print("qw kappa: <0 poor, 0-0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, 0.8-1.0 almost perfect")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    main(args.run_dir)
