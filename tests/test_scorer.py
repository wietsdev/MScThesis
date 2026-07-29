import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scorer import parse_mcq_response, score_mcq_response, normalize_cloze, score_cloze_response


# --- parse_mcq_response ---

def test_bare_lowercase():
    assert parse_mcq_response("c") == "c"

def test_bare_uppercase():
    assert parse_mcq_response("C") == "c"

def test_parenthesized():
    assert parse_mcq_response("(C)") == "c"

def test_natural_language():
    assert parse_mcq_response("The answer is c.") == "c"

def test_letter_with_justification():
    assert parse_mcq_response("c) because the snowflakes melt and acquire a liquid layer.") == "c"

def test_junk_is_unparseable():
    assert parse_mcq_response("I cannot determine the answer from the information given.") == "unparseable"


# --- score_mcq_response ---

def test_correct():
    parsed, correct = score_mcq_response("b", gold="b")
    assert parsed == "b"
    assert correct is True

def test_wrong():
    parsed, correct = score_mcq_response("d", gold="c")
    assert parsed == "d"
    assert correct is False

def test_unparseable_score():
    parsed, correct = score_mcq_response("I don't know.", gold="a")
    assert parsed == "unparseable"
    assert correct == "unparseable"


# --- normalize_cloze (markdown stripping) ---

def test_cloze_bold_markdown():
    assert normalize_cloze("**heterogeneous**") == "heterogeneous"

def test_cloze_italic_markdown():
    assert normalize_cloze("*heterogeneous*") == "heterogeneous"

def test_cloze_backtick():
    assert normalize_cloze("`design`") == "design"

def test_cloze_quoted():
    assert normalize_cloze('"design"') == "design"

def test_cloze_bold_scores_correct():
    _, correct = score_cloze_response("**heterogeneous**", gold="heterogeneous")
    assert correct is True
