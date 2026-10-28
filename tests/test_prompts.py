from grpo_reasoner.prompts import SYSTEM_PROMPT, build_messages


def test_prompt_is_fixed_and_contains_format():
    assert "<think>" in SYSTEM_PROMPT
    assert "</think>" in SYSTEM_PROMPT
    assert "<answer>" in SYSTEM_PROMPT
    assert "</answer>" in SYSTEM_PROMPT


def test_messages_include_question_and_system_prompt():
    messages = build_messages("  What is 2 + 2?  ")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "Problem: What is 2 + 2?"}
