import pytest

from grpo_reasoner.rewards import (
    completion_text,
    correctness_reward,
    format_reward,
    has_required_format,
    length_penalty_reward,
)

GOOD = "<think>18 - 9 = 9</think><answer>9</answer>"
WRONG = "<think>18 - 9 = 10</think><answer>10</answer>"


def test_completion_text_handles_strings_and_chat_messages():
    assert completion_text("hello") == "hello"
    assert completion_text({"role": "assistant", "content": "hello"}) == "hello"
    assert completion_text([{"role": "assistant", "content": "hel"}, {"content": "lo"}]) == "hello"


@pytest.mark.parametrize(
    ("completion", "gold", "expected"),
    [
        (GOOD, "9", 1.0),
        (WRONG, "9", 0.0),
        ("#### 9", "9", 1.0),
        ("The answer is 1,000.", "1000", 1.0),
        ("<answer>4.50</answer>", "4.5", 1.0),
        ("<answer>1/2</answer>", "0.5", 1.0),
        ("no number here", "9", 0.0),
        ([{"role": "assistant", "content": GOOD}], "9", 1.0),
    ],
)
def test_correctness_reward(completion, gold, expected):
    assert correctness_reward([completion], [gold]) == [expected]


def test_correctness_reward_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        correctness_reward(["a", "b"], ["1"])


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (GOOD, 1.0),
        ("  \n<think>reasoning</think>\n<answer>9</answer>\n ", 1.0),
        ([{"role": "assistant", "content": GOOD}], 1.0),
        ("<think>reasoning</think>", 0.0),
        ("<answer>9</answer>", 0.0),
        ("<answer>9</answer><think>reasoning</think>", 0.0),
        ("Before <think>reasoning</think><answer>9</answer>", 0.0),
        ("<think>reasoning</think> extra <answer>9</answer>", 0.0),
        ("<think>reasoning</think><answer>9</answer> after", 0.0),
        ("<think></think><answer>9</answer>", 0.0),
        ("<think>reasoning</think><answer>  </answer>", 0.0),
        ("<THINK>reasoning</THINK><ANSWER>9</ANSWER>", 0.0),
        (GOOD + GOOD, 0.0),
    ],
)
def test_format_reward(completion, expected):
    assert format_reward([completion]) == [expected]


def test_has_required_format_is_pure():
    text = "<think>x</think><answer>1</answer>"
    assert has_required_format(text) is True
    assert has_required_format(text) is True
    assert text == "<think>x</think><answer>1</answer>"


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("short", 0.0),
        ("x" * 2000, 0.0),
        ("x" * 2500, -0.025),
        ("x" * 4000, -0.1),
        ("x" * 8000, -0.1),
    ],
)
def test_length_penalty_reward(completion, expected):
    assert length_penalty_reward([completion], target_chars=2000, weight=0.1) == [expected]


def test_length_penalty_reward_custom_weight():
    assert length_penalty_reward(["x" * 3000], target_chars=2000, weight=0.2) == [-0.1]


def test_length_penalty_reward_validates_arguments():
    with pytest.raises(ValueError):
        length_penalty_reward(["x"], target_chars=0)
    with pytest.raises(ValueError):
        length_penalty_reward(["x"], weight=-0.1)
