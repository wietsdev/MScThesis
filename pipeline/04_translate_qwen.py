"""
Stage 4b: Translate the English master using Qwen3-235B via the gateway --
the cross-check translator role (NLLB-200 on Myriad is primary; see
cluster/translate_pilot.py). Per project decision, translator/judge/SUT
model roles must never overlap; qwen3-235b-generator is already used for MCQ
generation and is a different family from every SUT, so this reuses that
existing registry entry rather than adding a new one.

Unlike NLLB (per-sentence-fragment translation, batched), this translates a
whole item (question + all options together, or the claim, or the freeform
question) in ONE call via a JSON-structured prompt -- gateway calls aren't
batchable the way local HF inference is, and giving the model the full item
in context also tends to produce more coherent MCQ distractor translations
than translating each option in isolation.

Scope matches translate_pilot.py: evidence sentences are carried through
untranslated, English (same deliberate scope choice); gold labels/letters
are never translated, only option TEXT under the same fixed letter keys.

Reads:
    An english_master_v*.jsonl file (latest by default).

Writes:
    JSONL of translated Items: language=<lang>, translation_source="qwen3-235b-generator",
    translation_of=<original item_id>, item_id=<original item_id>_<lang>.
    Written incrementally with auto-resume (same pattern as
    pipeline/07_judge_freeform.py) -- safe to re-run after a kill/crash,
    picks up where it left off without re-paying for completed items.

Usage:
    uv run python pipeline/04_translate_qwen.py --lang pt --output data/full_pt_qwen.jsonl --n 5   # smoke test
    uv run python pipeline/04_translate_qwen.py --lang pt --output data/full_pt_qwen.jsonl          # full run
    uv run python pipeline/04_translate_qwen.py --lang hi --output data/full_hi_qwen.jsonl
    uv run python pipeline/04_translate_qwen.py --lang sw --output data/full_sw_qwen.jsonl
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from item_schema import Item

ROOT = Path(__file__).parent.parent
MODEL_ID = "qwen3-235b-generator"

LANG_NAMES = {
    "nl": "Dutch", "pt": "Portuguese", "hi": "Hindi", "sw": "Swahili",
    "zh": "Chinese (Simplified)", "es": "Spanish", "fr": "French",
    "ar": "Modern Standard Arabic", "bn": "Bengali", "id": "Indonesian",
    "ru": "Russian", "ur": "Urdu",
}

MCQ_PROMPT = """\
Translate the following climate-science multiple-choice question and its \
answer options into {lang_name}. Preserve technical/scientific terminology \
accurately. Translate each field independently -- do not add, remove, \
reorder, or explain anything.

QUESTION: {question}

OPTIONS:
{options_block}

Respond with valid JSON only (no markdown fences), using the SAME option \
letters as keys:
{{"question": "<translated question>", "options": {{{option_key_hints}}}}}
"""

CLAIM_PROMPT = """\
Translate the following climate-science claim into {lang_name}. Preserve \
technical/scientific terminology and the claim's exact meaning -- do not \
soften, hedge, or add commentary.

CLAIM: {claim}

Respond with valid JSON only (no markdown fences):
{{"claim": "<translated claim>"}}
"""

FREEFORM_PROMPT = """\
Translate the following climate-science question into {lang_name}. Preserve \
technical/scientific terminology and the question's exact meaning.

QUESTION: {question}

Respond with valid JSON only (no markdown fences):
{{"question": "<translated question>"}}
"""


def latest_master() -> Path:
    candidates = sorted(ROOT.glob("data/english_master_v*.jsonl"))
    if not candidates:
        print("No english_master_v*.jsonl found. Run pipeline/03_freeze.py first.")
        sys.exit(1)
    return candidates[-1]


# Observed failure mode translating into non-Latin scripts (confirmed on
# Hindi/Devanagari): the model frequently drops the OPENING quote around an
# option value -- e.g. `"c": ओजोन परत...",` instead of `"c": "ओजोन परत...",`
# -- sometimes the closing quote is still there, sometimes both are missing.
# Non-ASCII-aware ([^,}] includes Devanagari etc.) so it isn't Hindi-specific.
_MISSING_QUOTE_VALUE = re.compile(r'("[a-zA-Z]+"\s*:\s*)([^"\{\[\s][^,}]*?)(\s*[,}])')


def _repair_missing_quotes(text: str) -> str:
    def fix(m: re.Match) -> str:
        prefix, value, suffix = m.group(1), m.group(2).rstrip(), m.group(3)
        closing = "" if value.endswith('"') else '"'
        return f'{prefix}"{value}{closing}{suffix}'
    return _MISSING_QUOTE_VALUE.sub(fix, text)


def _parse_json(raw: str) -> dict | None:
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return json.loads(_repair_missing_quotes(text))
    except (json.JSONDecodeError, ValueError):
        return None


def build_prompt(item: Item, lang_name: str) -> str:
    if item.item_type == "mcq":
        letters = sorted(item.options.keys())
        options_block = "\n".join(f"{l}) {item.options[l]}" for l in letters)
        option_key_hints = ", ".join(f'"{l}": "<translated>"' for l in letters)
        return MCQ_PROMPT.format(lang_name=lang_name, question=item.question,
                                  options_block=options_block, option_key_hints=option_key_hints)
    if item.item_type == "claim":
        return CLAIM_PROMPT.format(lang_name=lang_name, claim=item.question)
    if item.item_type == "freeform":
        return FREEFORM_PROMPT.format(lang_name=lang_name, question=item.question)
    raise ValueError(f"Unknown item_type: {item.item_type!r}")


STRICT_JSON_SUFFIX = """\

IMPORTANT: your entire response must be a single, strictly valid JSON object -- \
nothing before or after it, no text outside the string values (e.g. no \
parenthetical attributions or notes appended after a closing quote), and \
every key correctly nested exactly as shown above."""


def translate_item(item: Item, lang: str, lang_name: str, dry_run: bool) -> Item | None:
    prompt = build_prompt(item, lang_name)

    if dry_run:
        result = {"question": f"[DRY RUN {lang}] {item.question}",
                   "options": dict(item.options) if item.options else None,
                   "claim": f"[DRY RUN {lang}] {item.question}"}
    else:
        import models
        # qwen3-235b-generator's registry default (max_tokens=800) is sized
        # for its original MCQ-generation use case. Devanagari (and likely
        # other non-Latin scripts) need more output tokens per character of
        # equivalent content than Latin scripts, so long items translated to
        # e.g. Hindi were getting cut off mid-JSON at exactly 800 tokens --
        # confirmed by observing truncated responses landing exactly at the
        # cap. 2000 covers even the longest observed item (a 956-char PIRA
        # option) with real margin, and the extra cost is negligible either way.
        raw = models.generate(prompt, MODEL_ID, temperature=0.0, max_tokens=2000)
        result = _parse_json(raw)
        if result is None:
            # temperature=0.0 makes a plain retry deterministic (same failure
            # again) -- a stricter instruction + a little temperature gives
            # the model an actual chance to produce parseable JSON instead.
            raw = models.generate(prompt + STRICT_JSON_SUFFIX, MODEL_ID, temperature=0.2, max_tokens=2000)
            result = _parse_json(raw)

    if result is None:
        return None

    if item.item_type == "mcq":
        translated_options = result.get("options")
        translated_question = result.get("question")
        if not translated_question or not translated_options or set(translated_options) != set(item.options):
            return None
    elif item.item_type == "claim":
        translated_question = result.get("claim")
        translated_options = None
        if not translated_question:
            return None
    else:  # freeform
        translated_question = result.get("question")
        translated_options = None
        if not translated_question:
            return None

    return Item(
        item_id=f"{item.item_id}_{lang}",
        source=item.source,
        license=item.license,
        item_type=item.item_type,
        language=lang,
        question=translated_question,
        gold=item.gold,
        options=translated_options,
        topic=item.topic,
        liu_topic=item.liu_topic,
        evidence=item.evidence,
        translation_source=MODEL_ID,
        translation_of=item.item_id,
    )


def main(input_path: Path, output_path: Path, lang: str, n: int | None,
         item_ids: list[str] | None, dry_run: bool) -> None:
    lang_name = LANG_NAMES[lang]
    all_items = [Item.from_dict(json.loads(l)) for l in input_path.read_text().splitlines() if l.strip()]

    if item_ids is not None:
        wanted = set(item_ids)
        items = [it for it in all_items if it.item_id in wanted]
        missing = wanted - {it.item_id for it in items}
        if missing:
            print(f"WARNING: {len(missing)} requested item_ids not found: {sorted(missing)[:10]}")
    elif n is not None:
        items = all_items[:n]
    else:
        items = all_items

    already_done: set[str] = set()
    if output_path.exists():
        for l in output_path.read_text().splitlines():
            if l.strip():
                already_done.add(json.loads(l)["translation_of"])

    remaining = [it for it in items if it.item_id not in already_done]
    if already_done:
        print(f"Resuming: {len(already_done)} already translated, {len(remaining)} remaining")

    print(f"Translating {len(remaining)} items -> {lang_name} ({lang}) via {MODEL_ID}")

    failed: list[str] = []
    t0 = time.time()
    with output_path.open("a") as f:
        for i, item in enumerate(remaining, 1):
            translated = translate_item(item, lang, lang_name, dry_run)
            if translated is None:
                failed.append(item.item_id)
            else:
                f.write(json.dumps(translated.to_dict()) + "\n")
                f.flush()
            if i % 25 == 0 or i == len(remaining):
                elapsed = time.time() - t0
                print(f"  {i}/{len(remaining)}  ({elapsed:.0f}s elapsed, {len(failed)} failed so far)")

    print(f"\nDone. Output: {output_path}")
    if failed:
        print(f"WARNING: {len(failed)} items failed to translate/parse (skipped, not written): {failed[:10]}"
              f"{'...' if len(failed) > 10 else ''}")
        print("Re-run the same command to retry only what's missing (resume is automatic).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None,
                        help="Master file to translate from (default: latest english_master_v*.jsonl)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lang", required=True, choices=list(LANG_NAMES),
                        help="Target language: pt (Portuguese), hi (Hindi), sw (Swahili)")
    parser.add_argument("--n", type=int, default=None, help="Only translate the first N items (smoke test)")
    parser.add_argument("--item-ids", type=Path, default=None, dest="item_ids_file",
                        help="Path to a text file, one item_id per line -- translate exactly these items")
    parser.add_argument("--dry-run", action="store_true", help="Skip gateway calls, write placeholder translations")
    args = parser.parse_args()
    item_ids = None
    if args.item_ids_file is not None:
        item_ids = [l.strip() for l in args.item_ids_file.read_text().splitlines() if l.strip()]
    main(args.input or latest_master(), args.output, args.lang, args.n, item_ids, args.dry_run)
