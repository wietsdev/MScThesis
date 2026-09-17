"""
Build an 8-rung translation quality ladder for a fixed item sample, so a
single SUT model can be run across all 8 rungs and accuracy plotted against
CometKiwi score -- testing how much translation quality has to degrade
before it actually shows up in downstream accuracy (not just a QE-score
sanity check).

Rungs (all covering the SAME items, from data/sonnet_sample_{lang}.jsonl's
translation_of ids):
  1_sonnet     -- claude-sonnet translation (real, already exists) -- the
                  BASE that every synthetic rung below post-edits, not a
                  fresh translation
  2_hybrid     -- current hybrid (NLLB+Qwen3, CometKiwi-selected winner) -- real
  3_loser      -- the hybrid's LOSING candidate for that item -- real, a
                  naturally-worse translation, reported separately (its
                  severity isn't controlled the way rungs 4-8 are, and it's
                  language-dependent: it was the worst rung for Swahili but
                  better than every synthetic rung for Dutch)
  4_degraded   -- rung 1, post-edited for worse fluency/word-choice only,
                  meaning unchanged
  5_negation   -- rung 1, post-edited: exactly one key claim's polarity flipped
  6_number     -- rung 1, post-edited: exactly one numeric value/unit swapped
  7_technical  -- rung 1, post-edited: exactly one technical term swapped for
                  a wrong-but-plausible one
  8_severe     -- rung 1, post-edited: fluency degraded AND 2-3 technical
                  terms wrong AND one meaning-changing flip, all at once --
                  the floor test

Design: rungs 4-8 POST-EDIT rung 1's own text (not translate fresh from
English). Translating fresh would confound two variables at once -- the
deliberate corruption, and the degrader model's own baseline translation
style/quality versus sonnet's. Editing rung 1 in place means every rung
differs from rung 1 by exactly the one intended change, nothing else.

Rungs 4-7 (single, surgical edits) use claude-haiku -- editing-in-place is an
easier task than generating a full translation, and haiku is proven reliable
on it. Rung 8 (three simultaneous constraints) uses claude-sonnet -- reliably
juggling multiple constraints without drifting off-spec is exactly where
stronger instruction-following earns its cost. Neither is gemma3-12b (the SUT
being tested) or either real translator, per the project's "translator/
judge/SUT roles must never overlap" rule; sonnet doing both rung 1 (a clean
translation) and rung 8 (an edit task) isn't a conflict since rung 8 is a
mechanical instruction-following task, not a quality judgment.

Every rung 4-8 item also gets an "edit_log" -- a companion file recording
exactly what the degrader model changed (field, original snippet, edited
snippet), NEVER included in the Item written for evaluation, so the SUT model
never sees it. This exists because a corruption instruction that says "don't
flag the change" is otherwise unverifiable at scale -- there was previously
no way to confirm the model actually complied or to know what error type
landed on which item.

Reads:
    data/english_master_v9.jsonl, data/sonnet_sample_{lang}.jsonl,
    data/full_{lang}_hybrid.jsonl, data/full_{lang}_nllb.jsonl,
    data/full_{lang}_qwen.jsonl

Writes:
    data/ladder_{lang}_1_sonnet.jsonl ... data/ladder_{lang}_8_severe.jsonl
    (Item-schema, ready for pipeline/06_run_eval.py --input)
    data/ladder_{lang}_{4..8}_{tag}_editlog.jsonl
    (item_id -> what was changed, for auditing rungs 4-8 -- never fed to eval)

Usage:
    uv run python scripts/build_quality_ladder.py --lang sw
    uv run python scripts/build_quality_ladder.py --lang sw --dry-run
    uv run python scripts/build_quality_ladder.py --lang sw --rungs 8
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from item_schema import Item

ROOT = Path(__file__).parent.parent
MASTER_PATH = ROOT / "data" / "english_master_v9.jsonl"

LANG_NAMES = {"nl": "Dutch", "sw": "Swahili", "hi": "Hindi", "ur": "Urdu", "pt": "Portuguese"}
# Dutch's raw NLLB file predates the full_{lang}_nllb.jsonl naming convention
# (an early pilot-run artifact) -- see pipeline/05_tq_score.py's LANG_FILES.
NLLB_FILE_OVERRIDES = {"nl": "full_nl_181423.jsonl"}

# rung_num -> (rung_key, model). Keys 4-8 are post-edit rungs.
SYNTHETIC_RUNGS = [
    (4, "degraded", "claude-haiku"),
    (5, "negation", "claude-haiku"),
    (6, "number", "claude-haiku"),
    (7, "technical", "claude-haiku"),
    (8, "severe", "claude-sonnet"),
]
ALL_RUNG_NUMS = [1, 2, 3, 4, 5, 6, 7, 8]

INSTRUCTIONS = {
    "degraded": (
        "Rewrite this translation to sound noticeably more awkward and less fluent -- "
        "imprecise word choices, minor grammatical errors, like a weaker MT system "
        "produced it. Do NOT change any facts, numbers, or meaning -- every piece of "
        "semantic content must stay identical, only the fluency/word-choice quality "
        "gets worse."
    ),
    "negation": (
        "Find ONE key claim in this translation and flip its polarity -- e.g. an "
        "affirmative statement becomes its negation, or vice versa (e.g. 'is caused "
        "by' -> 'is not caused by'). This must be the ONLY change: copy every other "
        "field exactly as given, unchanged."
    ),
    "number": (
        "Find ONE numeric value or unit in this translation and change it to a "
        "different, plausible-looking WRONG value. This must be the ONLY change: copy "
        "every other field exactly as given, unchanged. If there is genuinely no "
        "number or unit anywhere in the text, say so honestly in edit_log instead of "
        "forcing an unrelated change."
    ),
    "technical": (
        "Find ONE technical/scientific term in this translation and replace it with a "
        "different, plausible-sounding but INCORRECT technical term -- the kind of "
        "mistake a translator unfamiliar with the domain (but otherwise fluent) would "
        "make. This must be the ONLY change: copy every other field exactly as given, "
        "unchanged."
    ),
    "severe": (
        "Make this translation genuinely bad by applying THREE changes at once, all "
        "to the same text: (1) noticeably degrade the fluency/word-choice throughout "
        "(like a weak MT system), (2) replace 2-3 technical/scientific terms with "
        "different, plausible-sounding but INCORRECT ones, and (3) flip the polarity "
        "of one key claim OR swap one numeric value for a wrong one (whichever the "
        "text supports). Apply all three simultaneously -- this is deliberately the "
        "worst rung in the ladder."
    ),
}

STRICT_JSON_SUFFIX = """\

IMPORTANT: your entire response must be a single, strictly valid JSON object -- \
nothing before or after it, no text outside the string values, and every key \
correctly nested exactly as shown above."""

_MISSING_QUOTE_VALUE = re.compile(r'("[a-zA-Z_]+"\s*:\s*)([^"\{\[\s][^,}]*?)(\s*[,}])')


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


POST_EDIT_MCQ_PROMPT = """\
Below is an existing {lang_name} translation of a climate-science multiple-choice \
question. Your job is to EDIT this translation per the instruction below -- do not \
retranslate from scratch, and do not change anything the instruction doesn't ask for.

ENGLISH SOURCE (for reference only):
QUESTION: {en_question}
OPTIONS:
{en_options_block}

CURRENT {lang_name} TRANSLATION (edit this):
QUESTION: {question}
OPTIONS:
{options_block}

EDIT INSTRUCTION: {instruction}

Return valid JSON only (no markdown fences). Include EVERY field, copying over \
anything you didn't change exactly as given, using the SAME option letters as keys. \
Also include an edit_log field describing precisely what you changed -- this is for \
internal record-keeping only, it is never shown to anyone evaluating the translation, \
so be honest and precise:
{{"question": "<...>", "options": {{{option_key_hints}}}, "edit_log": {{"field_changed": \
"question or the option letter", "original_snippet": "<exact original text>", \
"edited_snippet": "<exact new text>"}}}}
"""

POST_EDIT_CLAIM_PROMPT = """\
Below is an existing {lang_name} translation of a climate-science claim. Your job is \
to EDIT this translation per the instruction below -- do not retranslate from scratch, \
and do not change anything the instruction doesn't ask for.

ENGLISH SOURCE (for reference only): {en_claim}

CURRENT {lang_name} TRANSLATION (edit this): {claim}

EDIT INSTRUCTION: {instruction}

Return valid JSON only (no markdown fences), plus an edit_log field describing \
precisely what you changed -- for internal record-keeping only, never shown to anyone \
evaluating the translation:
{{"claim": "<edited claim>", "edit_log": {{"field_changed": "claim", \
"original_snippet": "<exact original text>", "edited_snippet": "<exact new text>"}}}}
"""

POST_EDIT_FREEFORM_PROMPT = """\
Below is an existing {lang_name} translation of a climate-science question. Your job \
is to EDIT this translation per the instruction below -- do not retranslate from \
scratch, and do not change anything the instruction doesn't ask for.

ENGLISH SOURCE (for reference only): {en_question}

CURRENT {lang_name} TRANSLATION (edit this): {question}

EDIT INSTRUCTION: {instruction}

Return valid JSON only (no markdown fences), plus an edit_log field describing \
precisely what you changed -- for internal record-keeping only, never shown to anyone \
evaluating the translation:
{{"question": "<edited question>", "edit_log": {{"field_changed": "question", \
"original_snippet": "<exact original text>", "edited_snippet": "<exact new text>"}}}}
"""


def build_post_edit_prompt(source: Item, base: Item, lang_name: str, rung_key: str) -> str:
    instruction = INSTRUCTIONS[rung_key]
    if base.item_type == "mcq":
        letters = sorted(base.options.keys())
        en_options_block = "\n".join(f"{l}) {source.options[l]}" for l in letters)
        options_block = "\n".join(f"{l}) {base.options[l]}" for l in letters)
        option_key_hints = ", ".join(f'"{l}": "<...>"' for l in letters)
        return POST_EDIT_MCQ_PROMPT.format(
            lang_name=lang_name, en_question=source.question, en_options_block=en_options_block,
            question=base.question, options_block=options_block, instruction=instruction,
            option_key_hints=option_key_hints,
        )
    if base.item_type == "claim":
        return POST_EDIT_CLAIM_PROMPT.format(
            lang_name=lang_name, en_claim=source.question, claim=base.question, instruction=instruction,
        )
    return POST_EDIT_FREEFORM_PROMPT.format(
        lang_name=lang_name, en_question=source.question, question=base.question, instruction=instruction,
    )


def generate_rung(source: Item, base: Item, lang: str, lang_name: str, rung_key: str,
                   model_id: str, dry_run: bool) -> tuple[Item | None, dict | None]:
    prompt = build_post_edit_prompt(source, base, lang_name, rung_key)

    if dry_run:
        result = {
            "question": f"[DRY {rung_key}] {base.question}",
            "options": dict(base.options) if base.options else None,
            "claim": f"[DRY {rung_key}] {base.question}",
            "edit_log": {"field_changed": "question", "original_snippet": "(dry run)", "edited_snippet": "(dry run)"},
        }
    else:
        import models
        raw = models.generate(prompt, model_id, temperature=0.3, max_tokens=1500)
        result = _parse_json(raw)
        if result is None:
            raw = models.generate(prompt + STRICT_JSON_SUFFIX, model_id, temperature=0.4, max_tokens=1500)
            result = _parse_json(raw)

    if result is None:
        return None, None

    edit_log = result.get("edit_log")

    if base.item_type == "mcq":
        translated_question = result.get("question")
        translated_options = result.get("options")
        if not translated_question or not translated_options or set(translated_options) != set(base.options):
            return None, None
    elif base.item_type == "claim":
        translated_question = result.get("claim")
        translated_options = None
        if not translated_question:
            return None, None
    else:
        translated_question = result.get("question")
        translated_options = None
        if not translated_question:
            return None, None

    edited_item = Item(
        item_id=f"{source.item_id}_{lang}", source=source.source, license=source.license,
        item_type=source.item_type, language=lang, question=translated_question,
        gold=source.gold, options=translated_options, topic=source.topic,
        liu_topic=source.liu_topic, evidence=source.evidence,
        translation_source=f"{model_id}-{rung_key}-postedit", translation_of=source.item_id,
    )
    return edited_item, edit_log


def load_jsonl(path: Path) -> list[Item]:
    return [Item.from_dict(json.loads(l)) for l in path.read_text().splitlines() if l.strip()]


def main(lang: str, dry_run: bool, rungs: list[int] | None) -> None:
    wanted_rungs = set(rungs) if rungs else set(ALL_RUNG_NUMS)
    lang_name = LANG_NAMES[lang]
    master_by_id = {it.item_id: it for it in load_jsonl(MASTER_PATH)}
    sonnet_items = load_jsonl(ROOT / "data" / f"sonnet_sample_{lang}.jsonl")
    wanted_ids = [it.translation_of for it in sonnet_items]
    base_items = [master_by_id[i] for i in wanted_ids]
    sonnet_by_orig = {it.translation_of: it for it in sonnet_items}

    if 1 in wanted_rungs:
        out1 = ROOT / "data" / "ladder_" f"{lang}_1_sonnet.jsonl"
        out1.write_text("\n".join(json.dumps(it.to_dict()) for it in sonnet_items) + "\n")
        print(f"1_sonnet:   {len(sonnet_items)} items -> {out1}")

    if {2, 3} & wanted_rungs:
        hybrid_by_orig = {it.translation_of: it for it in load_jsonl(ROOT / "data" / f"full_{lang}_hybrid.jsonl")}

    if 2 in wanted_rungs:
        hybrid_items = [hybrid_by_orig[i] for i in wanted_ids if i in hybrid_by_orig]
        out2 = ROOT / "data" / f"ladder_{lang}_2_hybrid.jsonl"
        out2.write_text("\n".join(json.dumps(it.to_dict()) for it in hybrid_items) + "\n")
        print(f"2_hybrid:   {len(hybrid_items)} items -> {out2}")

    if 3 in wanted_rungs:
        nllb_path = ROOT / "data" / NLLB_FILE_OVERRIDES.get(lang, f"full_{lang}_nllb.jsonl")
        nllb_by_orig = {it.translation_of: it for it in load_jsonl(nllb_path)}
        qwen_by_orig = {it.translation_of: it for it in load_jsonl(ROOT / "data" / f"full_{lang}_qwen.jsonl")}
        loser_items = []
        for orig_id in wanted_ids:
            h = hybrid_by_orig.get(orig_id)
            if h is None:
                continue
            loser = qwen_by_orig.get(orig_id) if h.translation_source == "facebook/nllb-200-3.3B" else nllb_by_orig.get(orig_id)
            if loser is not None:
                loser_items.append(loser)
        out3 = ROOT / "data" / f"ladder_{lang}_3_loser.jsonl"
        out3.write_text("\n".join(json.dumps(it.to_dict()) for it in loser_items) + "\n")
        print(f"3_loser:    {len(loser_items)} items -> {out3}")

    for rung_num, rung_key, model_id in SYNTHETIC_RUNGS:
        if rung_num not in wanted_rungs:
            continue
        out_items, edit_logs, failed = [], [], []
        for source in base_items:
            base = sonnet_by_orig.get(source.item_id)
            if base is None:
                failed.append(source.item_id)
                continue
            edited_item, edit_log = generate_rung(source, base, lang, lang_name, rung_key, model_id, dry_run)
            if edited_item is None:
                failed.append(source.item_id)
                continue
            out_items.append(edited_item)
            edit_logs.append({"item_id": edited_item.item_id, "edit_log": edit_log})

        out_path = ROOT / "data" / f"ladder_{lang}_{rung_num}_{rung_key}.jsonl"
        out_path.write_text("\n".join(json.dumps(it.to_dict()) for it in out_items) + "\n")
        log_path = ROOT / "data" / f"ladder_{lang}_{rung_num}_{rung_key}_editlog.jsonl"
        log_path.write_text("\n".join(json.dumps(e) for e in edit_logs) + "\n")
        print(f"{rung_num}_{rung_key} ({model_id}): {len(out_items)}/{len(base_items)} items -> {out_path}"
              + (f"  (FAILED: {failed})" if failed else ""))
        print(f"  edit log -> {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=list(LANG_NAMES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rungs", type=int, nargs="+", choices=ALL_RUNG_NUMS, default=None,
                        help="Only (re)build these rung numbers (default: all)")
    args = parser.parse_args()
    main(args.lang, args.dry_run, args.rungs)
