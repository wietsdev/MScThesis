import json
import re
import sys
from pathlib import Path

VALID = {"a", "b", "c", "d"}


# --- MCQ ---

def parse_mcq_response(raw: str) -> str:
    text = raw.strip().lower()

    # pass 1: strip surrounding punctuation/brackets; check for bare letter
    cleaned = text.strip("()[].,;: \t\n")
    if cleaned in VALID:
        return cleaned

    # pass 2: find a standalone a/b/c/d not preceded by another letter
    match = re.search(r"(?<![a-z])([a-d])\b", text)
    if match:
        return match.group(1)

    return "unparseable"


def score_mcq_response(raw: str, gold: str) -> tuple[str, bool | str]:
    parsed = parse_mcq_response(raw)
    if parsed == "unparseable":
        return parsed, "unparseable"
    correct = parsed == gold.strip().lower()
    return parsed, correct


def score_mcq_file(responses_path: Path) -> None:
    records = _load(responses_path)
    scores, misses = [], []

    for rec in records:
        parsed, correct = score_mcq_response(rec["response"], rec["gold"])
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


# --- Dispatcher ---

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
