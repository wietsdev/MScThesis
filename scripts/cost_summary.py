"""Print a summary of API spend from .cache/cost_log.jsonl."""

import json
from collections import defaultdict
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / ".cache" / "cost_log.jsonl"


def main():
    if not LOG_PATH.exists():
        print("No cost log found. No model calls have been made yet.")
        return

    entries = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]

    if not entries:
        print("Cost log is empty.")
        return

    total_cost = sum(e["cost"] for e in entries)
    total_input = sum(e.get("input_tokens", 0) for e in entries)
    total_output = sum(e.get("output_tokens", 0) for e in entries)
    remaining = entries[-1]["remaining"]
    first_ts = entries[0]["ts"]
    last_ts = entries[-1]["ts"]

    calls_by_model = defaultdict(int)
    cost_by_model = defaultdict(float)
    input_by_model = defaultdict(int)
    output_by_model = defaultdict(int)
    for e in entries:
        calls_by_model[e["model_id"]] += 1
        cost_by_model[e["model_id"]] += e["cost"]
        input_by_model[e["model_id"]] += e.get("input_tokens", 0)
        output_by_model[e["model_id"]] += e.get("output_tokens", 0)

    print(f"Cost log: {LOG_PATH}")
    print(f"Period:   {first_ts}  ->  {last_ts}")
    print(f"Calls:    {len(entries)}")
    print(f"Total spend:      ${total_cost:.4f}")
    print(f"Remaining budget: ${remaining:.2f}")
    print(f"Total tokens:     {total_input + total_output:,}  ({total_input:,} in / {total_output:,} out)")
    print()
    print(f"{'Model':<20} {'Calls':>6} {'Cost':>10} {'Input tok':>10} {'Output tok':>11}")
    print("-" * 62)
    for model_id in sorted(cost_by_model):
        print(
            f"{model_id:<20} {calls_by_model[model_id]:>6} "
            f"${cost_by_model[model_id]:.4f} "
            f"{input_by_model[model_id]:>10,} "
            f"{output_by_model[model_id]:>11,}"
        )


if __name__ == "__main__":
    main()
