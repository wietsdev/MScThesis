"""
Matrix eval runner: samples N items per split (fixed seed), runs every model
on the same items, saves raw responses + scores to results/matrix_<ts>/.
"""

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset

import models
from scorer import score_file

MODELS = ["claude-haiku", "qwen3-32b", "llama3-8b", "ministral-3-8b"]
SPLITS = ["mcq", "cloze"]
N_ITEMS = None   # None = full dataset; integer = random sample with SEED
SEED = 42
DATASET = "climaqa"
COST_LOG = Path(__file__).parent.parent / ".cache" / "cost_log.jsonl"


def sample_indices(split_ds, n: int | None, seed: int) -> list[int]:
    if n is None:
        return list(range(len(split_ds)))
    rng = random.Random(seed)
    return sorted(rng.sample(range(len(split_ds)), n))


def build_prompt(row: dict, split: str) -> str:
    if split == "mcq":
        return (
            f"Question: {row['Question']}\n\n"
            f"Options:\n{row['Options']}\n\n"
            "Answer with ONLY the letter (a, b, c, or d)."
        )
    if split == "cloze":
        return (
            # "Complete the sentence by replacing <MASK> with the correct word or phrase.\n"
            # "Answer with ONLY the missing word or phrase.\n\n"
            # f"{row['Question']}"
            "Task: Fill in the missing word or technical term represented by <MASK> in the sentence.\n"
            "Rule: Respond with ONLY the single missing word or short term. Do NOT explain or write full sentences.\n\n"
            "--- Examples ---\n"
            "Sentence: The horizontal transport of atmospheric properties such as heat or moisture by the wind field is known as <MASK>.\n"
            "Answer: advection\n\n"
            "Sentence: Interannual climate variability across the tropical Pacific is heavily driven by the <MASK> cycle.\n"
            "Answer: ENSO\n\n"
            "Sentence: Strong jet streams and upper-level divergence are commonly observed near the <MASK> pressure surface.\n"
            "Answer: 200-hPa\n\n"
            "Sentence: Incoming solar radiation reaching the top of the Earth's atmosphere is referred to as <MASK>.\n"
            "Answer: insolation\n"
            "--- End Examples ---\n\n"
            f"Sentence: {row['Question']}\n"
            "Answer:"
        )
    raise ValueError(f"No prompt builder for split: '{split}'")


def run_model_split(
    matrix_dir: Path,
    model_id: str,
    split: str,
    ds,
    indices: list[int],
) -> Path:
    run_dir = matrix_dir / f"{model_id}_{split}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_id": model_id,
        "model_name": models.MODEL_REGISTRY[model_id]["model_name"],
        "dataset": DATASET,
        "split": split,
        "n_items": len(indices),
        "item_indices": indices,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    responses_path = run_dir / "responses.jsonl"
    with responses_path.open("w") as f:
        for idx in indices:
            row = ds[split][idx]
            prompt = build_prompt(row, split)
            response = models.generate(prompt, model_id)
            record = {
                "id": idx,                          # dataset row index -- the join key
                "question": row["Question"],
                "gold": row["Answer"],
                "complexity": row.get("Complexity"),
                "prompt": prompt,
                "response": response,
            }
            f.write(json.dumps(record) + "\n")

    return responses_path


def log_entries_since(n_before: int) -> list[dict]:
    if not COST_LOG.exists():
        return []
    lines = [l for l in COST_LOG.read_text().splitlines() if l.strip()]
    return [json.loads(l) for l in lines[n_before:]]


def main():
    print("Loading dataset...")
    ds = load_dataset("Rose-STL-Lab/ClimaQA", "Gold")

    indices = {split: sample_indices(ds[split], N_ITEMS, SEED) for split in SPLITS}

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    matrix_dir = Path(__file__).parent.parent / "results" / f"matrix_{ts}"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    matrix_config = {
        "ts": ts,
        "models": MODELS,
        "splits": SPLITS,
        "n_items": N_ITEMS,
        "seed": SEED,
        "dataset": DATASET,
        "item_indices": indices,
    }
    (matrix_dir / "matrix_config.json").write_text(json.dumps(matrix_config, indent=2))

    print(f"Matrix dir: {matrix_dir.name}")
    for split in SPLITS:
        n = len(indices[split])
        preview = str(indices[split]) if n <= 20 else f"[0..{indices[split][-1]}] ({n} items)"
        print(f"  {split}: {preview}")
    print()

    n_before = sum(
        1 for l in COST_LOG.read_text().splitlines() if l.strip()
    ) if COST_LOG.exists() else 0

    # --- calls ---
    total = len(MODELS) * len(SPLITS)
    n = 0
    for model_id in MODELS:
        for split in SPLITS:
            n += 1
            print(f"\n[{n}/{total}] {model_id} / {split}")
            run_model_split(matrix_dir, model_id, split, ds, indices[split])

    # --- scoring ---
    print("\n" + "=" * 50)
    print("Scoring all runs...")
    for model_id in MODELS:
        for split in SPLITS:
            responses_path = matrix_dir / f"{model_id}_{split}" / "responses.jsonl"
            print(f"\n{model_id} / {split}")
            score_file(responses_path)

    # --- cost summary ---
    new_entries = log_entries_since(n_before)
    if new_entries:
        run_cost = sum(e["cost"] for e in new_entries)
        run_tokens_in = sum(e.get("input_tokens", 0) for e in new_entries)
        run_tokens_out = sum(e.get("output_tokens", 0) for e in new_entries)
        remaining = new_entries[-1]["remaining"]
        print(f"\n{'=' * 50}")
        print(f"Matrix complete.  Results in: {matrix_dir.name}")
        print(f"This run:  ${run_cost:.4f}  |  {run_tokens_in + run_tokens_out:,} tokens  |  remaining: ${remaining:.2f}")
    else:
        print(f"\nMatrix complete (all responses served from cache).")
        print(f"Results in: {matrix_dir.name}")


if __name__ == "__main__":
    main()
