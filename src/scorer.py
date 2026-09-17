import json
import re
import sys
from pathlib import Path

VALID = {"a", "b", "c", "d"}   # default for 4-option MCQ; PIRA (5-option) passes its own set


# --- MCQ ---
# valid_letters is parameterized (not hardcoded a-d) because PIRA items have
# 5 options (a-e). Never aggregate accuracy across different option counts --
# score/report ClimaQA+generated (4-opt) and PIRA (5-opt) as separate groups.

def parse_mcq_response(raw: str, valid_letters: set[str] = VALID) -> str:
    text = raw.strip().lower()

    # pass 1: strip surrounding punctuation/brackets; check for bare letter
    cleaned = text.strip("()[].,;: \t\n")
    if cleaned in valid_letters:
        return cleaned

    # pass 2: find a standalone letter (from valid_letters) not preceded by another letter
    letter_class = "".join(sorted(valid_letters))
    match = re.search(rf"(?<![a-z])([{letter_class}])\b", text)
    if match:
        return match.group(1)

    return "unparseable"


def score_mcq_response(raw: str, gold: str, valid_letters: set[str] = VALID) -> tuple[str, bool | str]:
    parsed = parse_mcq_response(raw, valid_letters)
    if parsed == "unparseable":
        return parsed, "unparseable"
    correct = parsed == gold.strip().lower()
    return parsed, correct


def rotate_mcq_options(options: dict[str, str], gold: str, rotation: int) -> tuple[dict[str, str], str]:
    """Cyclically rotate option texts across the fixed letter slots (a, b, c, ...).

    Used for cyclic-permutation position-bias debiasing (pipeline/06_run_eval.py
    --debias-mcq): running an item under every rotation of its own options means
    the correct answer sits in every letter slot exactly once across the k runs,
    so a model's positional preference can't inflate its aggregate accuracy.
    rotation=0 returns the original arrangement unchanged.
    """
    letters = sorted(options.keys())
    texts = [options[l] for l in letters]
    k = len(letters)
    gold_idx = letters.index(gold)
    rotated = {letters[i]: texts[(i + rotation) % k] for i in range(k)}
    new_gold = letters[(gold_idx - rotation) % k]
    return rotated, new_gold


def aggregate_circular_mcq(runs: list[dict]) -> dict:
    """Combine the k rotation runs for one MCQ item into a single scored outcome.

    runs: one dict per rotation, each with 'correct' (bool | 'unparseable'),
    'gold' (that rotation's gold letter) and 'chosen_text' (the option text the
    model's parsed letter pointed to, or None if unparseable).

    'correct' is a majority vote across rotations (ties -- possible only for
    even k -- count as incorrect, since the model wasn't reliably right); this
    is what feeds the normal per-source accuracy summary. 'consistency' checks
    whether the model chose the SAME option content regardless of which letter
    it was shown under -- a model with zero real knowledge but a strong letter
    preference will look consistent-by-letter but inconsistent-by-content.
    """
    k = len(runs)
    n_correct = sum(1 for r in runs if r["correct"] is True)
    n_unparseable = sum(1 for r in runs if r["correct"] == "unparseable")

    if n_unparseable == k:
        correct: bool | str = "unparseable"
    else:
        correct = (n_correct * 2 > k)

    chosen_texts = {r["chosen_text"] for r in runs if r["chosen_text"] is not None}
    consistent = len(chosen_texts) == 1

    return {
        "correct": correct,
        "circular_accuracy": n_correct / k,
        "consistent": consistent,
        "n_permutations": k,
    }


def score_mcq_file(responses_path: Path) -> None:
    records = _load(responses_path)
    scores, misses = [], []

    for rec in records:
        valid_letters = set(rec["options"].keys()) if rec.get("options") else VALID
        parsed, correct = score_mcq_response(rec["response"], rec["gold"], valid_letters)
        scored = _scored_record(rec, parsed, correct)
        scores.append(scored)
        if correct is not True:
            misses.append(scored)

    _write_and_summarise(responses_path, scores, misses)


# --- Claim (Climate-FEVER: 4-class exact match) ---

CLAIM_LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]


def parse_claim_response(raw: str) -> str:
    text = raw.strip().upper()

    # pass 1: exact match after stripping punctuation
    cleaned = text.strip("()[].,;:!\"' \t\n")
    if cleaned in CLAIM_LABELS:
        return cleaned

    # pass 2: the model wrote a sentence -- find a labelled whole word/phrase in it.
    # Check NOT_ENOUGH_INFO's near-synonym phrasing too, since models often say
    # "not enough information" instead of the exact underscored label.
    if re.search(r"\bNOT[ _]ENOUGH[ _]INFO(RMATION)?\b", text):
        return "NOT_ENOUGH_INFO"
    for label in ("SUPPORTS", "REFUTES", "DISPUTED"):
        if re.search(rf"\b{label}\b", text):
            return label

    return "unparseable"


def score_claim_response(raw: str, gold: str) -> tuple[str, bool | str]:
    parsed = parse_claim_response(raw)
    if parsed == "unparseable":
        return parsed, "unparseable"
    correct = parsed == gold.strip().upper()
    return parsed, correct


def score_claim_file(responses_path: Path) -> None:
    records = _load(responses_path)
    scores, misses = [], []

    for rec in records:
        parsed, correct = score_claim_response(rec["response"], rec["gold"])
        scored = _scored_record(rec, parsed, correct)
        scores.append(scored)
        if correct is not True:
            misses.append(scored)

    _write_and_summarise(responses_path, scores, misses)


# --- Cloze ---

def normalize_cloze(text: str) -> str:
    # strip markdown emphasis and backtick formatting before anything else
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # **bold**
    text = re.sub(r"\*(.+?)\*",     r"\1", text)   # *italic*
    text = re.sub(r"__(.+?)__",     r"\1", text)   # __bold__
    text = re.sub(r"_(.+?)_",       r"\1", text)   # _italic_
    text = re.sub(r"`(.+?)`",       r"\1", text)   # `code`
    return text.strip().lower().strip("\"'.,;:!?()- \t\n")


def parse_cloze_response(raw: str) -> str:
    normalized = normalize_cloze(raw)
    return normalized if normalized else "unparseable"


def score_cloze_response(raw: str, gold: str) -> tuple[str, bool | str]:
    parsed = parse_cloze_response(raw)
    if parsed == "unparseable":
        return parsed, "unparseable"
    correct = parsed == normalize_cloze(gold)
    return parsed, correct


def score_cloze_file(responses_path: Path) -> None:
    records = _load(responses_path)
    scores, misses = [], []

    for rec in records:
        parsed, correct = score_cloze_response(rec["response"], rec["gold"])
        scored = _scored_record(rec, parsed, correct)
        scores.append(scored)
        # every miss goes to the miss log -- synonyms/morphology need eyeballing
        if correct is not True:
            misses.append(scored)

    _write_and_summarise(responses_path, scores, misses)


# --- Unified master-schema scoring (pipeline/06_run_eval.py) ---
# Records here are shaped like the Item schema (item_id, source, item_type,
# question, gold, options, topic, response) -- NOT the old {id, split,
# complexity} shape the score_file() dispatcher above expects. Grouped by
# `source` (not item_type) so 4-option (ClimaQA/generated) and 5-option
# (PIRA) MCQ are always reported separately, per the never-aggregate rule.

def score_master_record(item: dict) -> dict:
    """Score one master-schema record (with a 'response' field already set). Returns the scored dict."""
    item_type = item["item_type"]
    if item_type == "mcq":
        valid_letters = set(item["options"].keys())
        parsed, correct = score_mcq_response(item["response"], item["gold"], valid_letters)
    elif item_type == "claim":
        parsed, correct = score_claim_response(item["response"], item["gold"])
    elif item_type == "freeform":
        # No gold answer to match against -- freeform stays unscored until an
        # LLM judge is built (per project decision). Response is still kept.
        parsed, correct = item["response"], None
    else:
        raise ValueError(f"Unknown item_type: {item_type!r}")

    return {
        "item_id":  item["item_id"],
        "source":   item["source"],
        "item_type": item_type,
        "topic":    item.get("topic", []),
        "question": item["question"],
        "gold":     item["gold"],
        "raw_response": item["response"],
        "parsed":   parsed,
        "correct":  correct,   # True / False / "unparseable" / None (freeform, unscored)
    }


def summarise_by_source(scored: list[dict]) -> dict:
    """Group scored master-schema records by source; compute accuracy per group (freeform excluded)."""
    by_source: dict[str, list[dict]] = {}
    for s in scored:
        by_source.setdefault(s["source"], []).append(s)

    summary = {}
    for source, records in by_source.items():
        item_type = records[0]["item_type"]
        if item_type == "freeform":
            summary[source] = {
                "item_type": item_type, "n": len(records),
                "scored": "unscored (pending LLM judge)",
            }
            continue
        n_correct = sum(1 for r in records if r["correct"] is True)
        n_wrong = sum(1 for r in records if r["correct"] is False)
        n_unparseable = sum(1 for r in records if r["correct"] == "unparseable")
        n_total = len(records)
        summary[source] = {
            "item_type": item_type, "n": n_total,
            "correct": n_correct, "wrong": n_wrong, "unparseable": n_unparseable,
            "accuracy": n_correct / n_total if n_total else 0.0,
        }
    return summary


# --- Dispatcher (old {id, split, complexity}-shaped records; run_matrix.py / run_eval.py) ---

def score_file(responses_path: Path) -> None:
    config_path = responses_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {responses_path.parent}")
    split = json.loads(config_path.read_text())["split"]
    if split == "mcq":
        score_mcq_file(responses_path)
    elif split == "cloze":
        score_cloze_file(responses_path)
    else:
        raise ValueError(f"No scorer for split: '{split}'")


# --- Shared helpers ---

def _load(responses_path: Path) -> list[dict]:
    lines = [l for l in responses_path.read_text().splitlines() if l.strip()]
    return [json.loads(l) for l in lines]


def _scored_record(rec: dict, parsed: str, correct: bool | str) -> dict:
    return {
        "id": rec["id"],
        "question": rec["question"],
        "gold": rec["gold"],
        "complexity": rec.get("complexity"),
        "raw_response": rec["response"],
        "parsed": parsed,
        "correct": correct,
    }


def _write_and_summarise(responses_path: Path, scores: list, misses: list) -> None:
    run_dir = responses_path.parent
    scores_path = run_dir / "scores.jsonl"
    misses_path = run_dir / "misses.jsonl"

    scores_path.write_text("\n".join(json.dumps(s) for s in scores) + "\n")
    if misses:
        misses_path.write_text("\n".join(json.dumps(m) for m in misses) + "\n")

    n_total = len(scores)
    n_correct = sum(1 for s in scores if s["correct"] is True)
    n_wrong = sum(1 for s in scores if s["correct"] is False)
    n_unparseable = sum(1 for s in scores if s["correct"] == "unparseable")
    accuracy = n_correct / n_total if n_total else 0.0

    print(f"Scored:       {responses_path}")
    print(f"Total:        {n_total}")
    print(f"Correct:      {n_correct}")
    print(f"Wrong:        {n_wrong}")
    print(f"Unparseable:  {n_unparseable}")
    print(f"Accuracy:     {accuracy:.1%}")
    print(f"Scores ->     {scores_path}")
    if misses:
        print(f"Misses ->     {misses_path}  ({len(misses)} items)")
    else:
        print("No misses.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python src/scorer.py <path_to_responses.jsonl>")
        sys.exit(1)
    score_file(Path(sys.argv[1]))
