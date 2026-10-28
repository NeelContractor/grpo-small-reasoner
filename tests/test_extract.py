import pytest

from grpo_reasoner.extract import answers_match, extract_answer, extract_gold_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>18 - 9 = 9</think><answer>9</answer>", "9"),
        ("<answer>\n18\n</answer>", "18"),
        ("<answer>The answer is 1,234.</answer>", "1234"),
        ("<answer>$42</answer>", "42"),
        ("<answer>-7</answer>", "-7"),
        ("<answer>\u22125</answer>", "-5"),
        ("<answer>0.50</answer>", "0.5"),
        ("<answer>3.50</answer>", "3.5"),
        ("<answer>1/2</answer>", "1/2"),
        ("<answer>18</answer> I also thought about 21.", "18"),
        ("<answer>7</answer><answer>8</answer>", "8"),
        ("#### 42", "42"),
        ("Some reasoning.\n#### 1,000", "1000"),
        ("So the answer is 3.50", "3.5"),
        ("Work\nTherefore, the answer is -7.", "-7"),
        ("Step one gives 2\nStep two gives 3\nTotal: 5", "5"),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "I do not know.",
        "<answer></answer> 12",
        "<think>reasoning without numerals</think>",
    ],
)
def test_extract_answer_returns_none(text):
    assert extract_answer(text) is None


def test_answer_tag_takes_priority_over_later_numbers():
    assert extract_answer("<answer>10</answer>\nThen I checked 99.") == "10"


def test_gold_extraction():
    reference = "Natalia sold 48 clips in April.\n#### 48"
    assert extract_gold_answer(reference) == "48"


def test_gold_extraction_without_marker():
    assert extract_gold_answer("The total is 12.") == "12"


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        ("18", "18", True),
        ("1,000", "1000", True),
        ("18.0", "18", True),
        ("1/2", "0.5", True),
        ("$42", "42", True),
        ("-5", "-5", True),
        ("19", "18", False),
        ("1/3", "0.5", False),
        (None, "18", False),
        ("18", None, False),
        ("no number", "18", False),
    ],
)
def test_answers_match(prediction, gold, expected):
    assert answers_match(prediction, gold) is expected
