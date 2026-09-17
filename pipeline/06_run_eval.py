"""
Stage 6: Run one SUT model over the frozen English master and score it.

Handles all four item shapes in the master in one pass:
  mcq (4-option: ClimaQA/generated_mcq; 5-option: PIRA) -- exact-match on
    the answer letter, scored with valid_letters taken from that item's own
    options dict so 4- and 5-option items are never confused with each other.
  claim (Climate-FEVER) -- 4-class exact match (SUPPORTS/REFUTES/
    NOT_ENOUGH_INFO/DISPUTED).
  freeform (CLINB) -- response collected, NOT scored (no LLM judge yet --
    this is a deliberate decision already made, not a gap to fix here).

Results are grouped and reported by `source`, never aggregated across
sources -- ClimaQA (4-opt) and PIRA (5-opt) accuracy must stay separate
even though both are "mcq", per the project's own scoring rule.

Reads:
    data/english_master_v{N}.jsonl  (latest by default, or --input)

Writes:
    results/<run_id>/config.json
    results/<run_id>/responses.jsonl   raw item + response, one per line (one
        line per cyclic rotation for debiased MCQ items -- see below)
    results/<run_id>/scores.jsonl      scored records (see scorer.score_master_record;
        debiased MCQ items are one aggregated record per item, not one per rotation)
    results/<run_id>/summary.json      per-source accuracy breakdown
    results/<run_id>/mcq_position_bias.json   only with --debias-mcq: accuracy
        conditioned on which letter the correct answer occupies, grouped by
        option count -- generated_mcq only (see below), so the 4-opt group
        here is generated_mcq alone, and the 5-opt group is always empty
        (PIRA is never debiased)

--debias-mcq runs every generated_mcq item once per cyclic rotation of its own
options (see scorer.rotate_mcq_options) instead of once, so the correct answer
sits in every letter slot exactly once per item. This cancels a model's
positional preference in the reported accuracy (majority vote across
rotations) AND measures that preference directly via the position-bias table.
Costs 4x the normal MCQ call volume for generated_mcq -- opt in, not the
default. Scoped to generated_mcq ONLY, not ClimaQA-Gold or PIRA 2.0 -- those
are standard external benchmarks, scored the normal single-pass way so results
stay comparable to how those datasets are reported elsewhere (project
decision, 2026-08-27; see the scoring loop in main() for the exact gate).

Usage:
    uv run python pipeline/06_run_eval.py --model gemma3-12b --n 40   # pilot
    uv run python pipeline/06_run_eval.py --model gemma3-12b          # full master
    uv run python pipeline/06_run_eval.py --model gemma3-12b --debias-mcq
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import models
from scorer import (
    aggregate_circular_mcq,
    rotate_mcq_options,
    score_master_record,
    score_mcq_response,
    summarise_by_source,
)

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

SEED = 42


def latest_master() -> Path:
    candidates = sorted(ROOT.glob("data/english_master_v*.jsonl"))
    if not candidates:
        print("No english_master_v*.jsonl found. Run pipeline/03_freeze.py first.")
        sys.exit(1)
    return candidates[-1]


def build_prompt(item: dict) -> str:
    item_type = item["item_type"]

    if item_type == "mcq":
        letters = sorted(item["options"].keys())
        options_block = "\n".join(f"{l}) {item['options'][l]}" for l in letters)
        letters_str = "/".join(letters)
        return (
            f"Question: {item['question']}\n\n"
            f"Options:\n{options_block}\n\n"
            f"Answer with ONLY the letter ({letters_str})."
        )

    if item_type == "claim":
        # Climate-FEVER's own task is evidence-based verification -- DISPUTED
        # specifically means the evidence sentences disagree with each other,
        # which is unanswerable without seeing them. A quick closed-book-vs-
        # evidence-based pilot (gemma3-12b, n=12) showed 25% -> 58.3% accuracy
        # once evidence was included.
        evidence = item.get("evidence")
        if evidence:
            evidence_block = "\n".join(f"- {e['text']}" for e in evidence)
            return (
                f"Claim: {item['question']}\n\n"
                f"Evidence:\n{evidence_block}\n\n"
                "Based ONLY on the evidence above, classify the claim as exactly one of:\n"
                "- SUPPORTS: the evidence supports the claim\n"
                "- REFUTES: the evidence contradicts the claim\n"
                "- NOT_ENOUGH_INFO: the evidence doesn't say enough to judge the claim either way\n"
                "- DISPUTED: the evidence sentences disagree with each other about the claim\n\n"
                "Answer with ONLY the label."
            )
        # fallback for any claim item without evidence (shouldn't happen for
        # Climate-FEVER after pipeline/01_select.py's evidence retention, but
        # keeps this runner usable for a future claim source that lacks it)
        return (
            f"Claim: {item['question']}\n\n"
            "Classify this claim as exactly one of: SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED.\n"
            "Answer with ONLY the label."
        )

    if item_type == "freeform":
        # No length cap on the SUT side would let a verbose model (observed:
        # gemma3-12b) run past any reasonable token budget without ever
        # reaching a natural stopping point, truncating mid-sentence and
        # confounding the freeform judge's completeness scoring with a pure
        # token-budget artifact. Bounding it here is also more representative
        # of real assistant usage than an unbounded essay.
        return (
            f"{item['question']}\n\n"
            "Answer clearly and completely in a well-organized response of "
            "no more than about 350 words."
        )

    raise ValueError(f"Unknown item_type: {item_type!r}")


def run_mcq_debiased(item: dict, model_id: str, dry_run: bool, f) -> tuple[dict, dict]:
    """Run one MCQ item under every cyclic rotation of its own options (see
    scorer.rotate_mcq_options). Writes one responses.jsonl line per rotation
    and returns (aggregated scored record, position-bucketed correctness
    for the global bias table -- {gold_letter: [bool, ...]})."""
    options, gold = item["options"], item["gold"]
    k = len(options)
    runs = []
    position_hits: dict[str, list[bool]] = {}

    for rotation in range(k):
        rot_options, rot_gold = rotate_mcq_options(options, gold, rotation)
        rot_item = {**item, "options": rot_options, "gold": rot_gold}
        prompt = build_prompt(rot_item)
        response = "[DRY RUN]" if dry_run else models.generate(prompt, model_id)
        parsed, correct = score_mcq_response(response, rot_gold, set(rot_options.keys()))
        chosen_text = rot_options.get(parsed)

        f.write(json.dumps({
            **rot_item, "prompt": prompt, "response": response,
            "rotation": rotation, "circular_group": item["item_id"],
        }) + "\n")
        runs.append({"correct": correct, "gold": rot_gold, "chosen_text": chosen_text})
        position_hits.setdefault(rot_gold, []).append(correct is True)

    agg = aggregate_circular_mcq(runs)
    record = {
        "item_id": item["item_id"], "source": item["source"], "item_type": "mcq",
        "topic": item.get("topic", []), "question": item["question"], "gold": gold,
        "raw_response": None, "parsed": None, **agg,
    }
    return record, position_hits


def main(model_id: str, input_path: Path, n: int | None, seed: int, dry_run: bool, debias_mcq: bool) -> None:
    items = [json.loads(l) for l in input_path.read_text().splitlines() if l.strip()]

    if n is not None:
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(items)), min(n, len(items))))
        items = [items[i] for i in idx]

    # Directory name includes the item language, not just timestamp+model --
    # confirmed by a real incident: launching the same model on several
    # DIFFERENT language files as near-simultaneous background processes let
    # two runs land on the same to-the-second timestamp, collide on the same
    # directory name, and silently scramble each other's responses.jsonl
    # (duplicate item_ids, malformed concatenated JSON lines) rather than
    # erroring -- one run (Hindi) never got its own directory at all.
    # Read from the data's own `language` field rather than the input
    # filename -- input files aren't named consistently (full_nl_hybrid vs
    # english_master_v9 vs full_pt_nllb), but every Item already carries its
    # own language code regardless of what the file happens to be called.
    # The exist_ok loop below is defense-in-depth against the residual case
    # of the exact same model+language relaunched within the same second.
    lang_code = items[0].get("language", "unk") if items else "unk"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_name = f"{ts}_{model_id}_{lang_code}_{'pilot' if n else 'full'}"
    run_dir = RESULTS_DIR / base_name
    suffix = 2
    # mkdir(exist_ok=False) is the atomic check -- a separate exists()-then-
    # mkdir() has a TOCTOU race: two processes launched in the same second
    # (same base_name) can both see "doesn't exist" for the same suffix and
    # both attempt to create it, and one raises FileExistsError. Confirmed as
    # a real failure, not theoretical: 2 of 4 same-language ladder-eval jobs
    # crashed this way when launched concurrently via xargs -P4. Catching the
    # exception and retrying is race-free because mkdir's existence check and
    # creation are a single atomic OS call.
    while True:
        try:
            run_dir.mkdir(parents=True)
            break
        except FileExistsError:
            run_dir = RESULTS_DIR / f"{base_name}-{suffix}"
            suffix += 1

    by_source_n = {}
    for it in items:
        by_source_n[it["source"]] = by_source_n.get(it["source"], 0) + 1

    config = {
        "ts": ts, "model_id": model_id,
        "model_name": models.MODEL_REGISTRY[model_id]["model_name"],
        "master_file": input_path.name, "n_items": len(items),
        "seed": seed, "by_source": by_source_n, "debias_mcq": debias_mcq,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Plain-text invocation record -- the exact command line and resolved
    # data file, so a results dir is self-explanatory without cross
    # -referencing config.json or guessing from the directory name. Added
    # after the run-dir naming collision incident: more provenance per run
    # makes this kind of mixup obvious immediately instead of needing a
    # forensic responses.jsonl audit to catch it.
    invocation = (
        f"command:    {' '.join(sys.argv)}\n"
        f"input file: {input_path.resolve()}\n"
        f"model_id:   {model_id}\n"
        f"n_items:    {len(items)}\n"
        f"debias_mcq: {debias_mcq}\n"
        f"dry_run:    {dry_run}\n"
        f"started_at: {ts}\n"
    )
    (run_dir / "invocation.txt").write_text(invocation)

    print(f"Run dir: {run_dir.name}")
    print(f"Model:   {model_id}")
    print(f"Master:  {input_path.name}  ({len(items)} items, by source: {by_source_n})")
    if debias_mcq:
        print("MCQ debiasing: ON, generated_mcq only -- every generated_mcq item runs once "
              "per cyclic option rotation (4x). ClimaQA-Gold/PIRA 2.0 stay single-pass.")
    print()

    responses_path = run_dir / "responses.jsonl"
    scores_path = run_dir / "scores.jsonl"
    scored_records = []
    # keyed by n_options (4 vs 5 -- never mix option-count groups, same rule as
    # source-level accuracy) -> {letter: [bool, ...]}, only populated when
    # debias_mcq is on
    position_bias: dict[int, dict[str, list[bool]]] = {}

    # Both responses.jsonl and scores.jsonl are written incrementally (flushed
    # per item) rather than buffered and written once at the end -- a
    # kill/crash partway through a long run (real risk: a 652-item debiased
    # run is ~1600+ SUT calls) should only lose the in-flight item, not every
    # response and score already collected.
    with responses_path.open("w") as f, scores_path.open("w") as scores_f:
        for i, item in enumerate(items, 1):
            # Debiasing is scoped to generated_mcq only -- ClimaQA-Gold and PIRA
            # 2.0 are standard external benchmarks, scored the normal (single-pass)
            # way so results stay comparable to how those datasets are reported
            # elsewhere. generated_mcq is the project's own synthetic set with no
            # external "standard" protocol to match, so it keeps the full
            # cyclic-rotation treatment. Project decision, 2026-08-27.
            if debias_mcq and item["item_type"] == "mcq" and item["source"] == "generated_mcq":
                record, position_hits = run_mcq_debiased(item, model_id, dry_run, f)
                scored_records.append(record)
                scores_f.write(json.dumps(record) + "\n")
                scores_f.flush()
                bucket = position_bias.setdefault(len(item["options"]), {})
                for letter, hits in position_hits.items():
                    bucket.setdefault(letter, []).extend(hits)
                if i % 25 == 0 or i == len(items):
                    print(f"  {i}/{len(items)}")
                continue

            prompt = build_prompt(item)
            # mcq/claim need only a letter/label -- the registry's short default
            # (sized for that) is fine. freeform needs real room for a genuine
            # answer, or every CLINB response comes back truncated mid-sentence,
            # which is useless for the LLM judge this is being collected for.
            # 500 wasn't enough -- gemma3-12b in particular writes long structured
            # answers and was getting cut off mid-sentence, which the judge then
            # (correctly, but confusingly) penalized as an incompleteness failure
            # that was really just a token-budget artifact.
            # 1500 (was 700): confirmed by direct testing that reasoning-heavy
            # frontier models (e.g. claude-opus-5) spend real tokens on
            # invisible internal reasoning before any visible answer text --
            # at 700, a freeform response hit the cap exactly (truncated
            # mid-answer) because reasoning tokens ate into the budget meant
            # for the ~350-word visible response. 1500 gives real headroom
            # for that on top of the original gemma3-12b-motivated margin
            # above the prompt's own ~350-word cap. Cheap for models that
            # don't need it -- max_tokens is a ceiling, not a target.
            gen_kwargs = {"max_tokens": 1500} if item["item_type"] == "freeform" else {}
            response = "[DRY RUN]" if dry_run else models.generate(prompt, model_id, **gen_kwargs)
            record = {**item, "prompt": prompt, "response": response}
            f.write(json.dumps(record) + "\n")
            f.flush()
            scored = score_master_record(record)
            scored_records.append(scored)
            scores_f.write(json.dumps(scored) + "\n")
            scores_f.flush()
            if i % 25 == 0 or i == len(items):
                print(f"  {i}/{len(items)}")

    summary = summarise_by_source(scored_records)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if debias_mcq:
        position_summary = {}
        for n_options, letters in position_bias.items():
            group_key = f"{n_options}-option"
            position_summary[group_key] = {
                letter: {
                    "n": len(hits), "correct": sum(hits),
                    "accuracy": sum(hits) / len(hits) if hits else 0.0,
                }
                for letter, hits in sorted(letters.items())
            }
        (run_dir / "mcq_position_bias.json").write_text(json.dumps(position_summary, indent=2))

    print(f"\n{'─'*60}")
    print(f"Results: {run_dir}")
    print(f"\n{'Source':<18} {'type':<10} {'n':>5}  {'accuracy':>9}  (correct/wrong/unparseable)")
    for source, s in summary.items():
        if s["item_type"] == "freeform":
            print(f"{source:<18} {s['item_type']:<10} {s['n']:>5}  {'unscored':>9}  (pending LLM judge)")
        else:
            print(f"{source:<18} {s['item_type']:<10} {s['n']:>5}  {s['accuracy']:>8.1%}  "
                  f"({s['correct']}/{s['wrong']}/{s['unparseable']})")

    if debias_mcq:
        print(f"\n{'─'*60}")
        print("Position-bias diagnostic (accuracy conditioned on which letter the")
        print("correct answer currently occupies -- a flat row means no positional")
        print("shortcut; a skewed row means the model favours/avoids that letter):")
        for group_key, letters in position_summary.items():
            print(f"\n  {group_key}:")
            for letter, s in letters.items():
                print(f"    {letter}: {s['accuracy']:>6.1%}  ({s['correct']}/{s['n']})")
        n_consistent = sum(1 for r in scored_records if r.get("consistent") is True)
        n_mcq_debiased = sum(1 for r in scored_records if "consistent" in r)
        if n_mcq_debiased:
            print(f"\n  Content-consistency across rotations: {n_consistent}/{n_mcq_debiased} "
                  f"({n_consistent / n_mcq_debiased:.1%}) -- chose the same option content "
                  "regardless of which letter it was shown under")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID from src/models.py MODEL_REGISTRY")
    parser.add_argument("--input", type=Path, default=None,
                        help="Master file to evaluate (default: latest english_master_v*.jsonl)")
    parser.add_argument("--n", type=int, default=None,
                        help="Random sample size for a pilot run (default: full master)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls, write placeholder responses")
    parser.add_argument("--debias-mcq", action="store_true",
                         help="Run every generated_mcq item once per cyclic option rotation (4x "
                              "cost) to cancel and measure position bias. Scoped to generated_mcq "
                              "only -- ClimaQA-Gold and PIRA 2.0 are scored single-pass to stay "
                              "comparable to how those benchmarks are reported elsewhere. See "
                              "scorer.rotate_mcq_options / aggregate_circular_mcq.")
    args = parser.parse_args()
    main(args.model, args.input or latest_master(), args.n, args.seed, args.dry_run, args.debias_mcq)
