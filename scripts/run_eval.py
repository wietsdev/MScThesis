"""
Dry-run eval script: loads N items from ClimaQA-Gold, calls one model,
saves raw responses to results/. No scoring yet.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset

import models

MODEL_ID = "claude-haiku"
SPLIT = "cloze"   # "mcq" or "cloze"
N_ITEMS = 5
DATASET = "climaqa"


def simple_mcq_prompt(row: dict) -> str:
    return (
        f"Question: {row['Question']}\n\n"
        f"Options:\n{row['Options']}\n\n"
        "Answer with only the letter (a, b, c, or d)."
    )


def simple_cloze_prompt(row: dict) -> str:
    return (
        f"Complete the sentence by replacing <MASK> with the correct word or phrase.\n"
        f"Answer with only the missing word or phrase, nothing else.\n\n"
        f"{row['Question']}"
    )


def build_prompt(row: dict, split: str) -> str:
    if split == "mcq":
        return simple_mcq_prompt(row)
    if split == "cloze":
        return simple_cloze_prompt(row)
    raise ValueError(f"No prompt builder for split: '{split}'")


def main():
    print("Loading dataset...")
    ds = load_dataset("Rose-STL-Lab/ClimaQA", "Gold")
    items = list(ds[SPLIT].select(range(N_ITEMS)))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = (
        Path(__file__).parent.parent
        / "results"
        / f"{ts}_{MODEL_ID}_{DATASET}_{SPLIT}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ts": ts,
        "model_id": MODEL_ID,
        "model_name": models.MODEL_REGISTRY[MODEL_ID]["model_name"],
        "dataset": DATASET,
        "split": SPLIT,
        "n_items": N_ITEMS,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Run dir: {run_dir.name}")
    print(f"Model:   {MODEL_ID}\n")

    responses_path = run_dir / "responses.jsonl"
    with responses_path.open("w") as f:
        for i, item in enumerate(items):
            prompt = build_prompt(item, SPLIT)
            print(f"[{i}] {item['Question'][:80]}...")
            response = models.generate(prompt, MODEL_ID)
            print(f"     response: {response!r}\n")

            record = {
                "id": i,
                "question": item["Question"],
                "gold": item["Answer"],
                "complexity": item["Complexity"],
                "prompt": prompt,
                "response": response,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {N_ITEMS} responses to {responses_path}")


if __name__ == "__main__":
    main()
