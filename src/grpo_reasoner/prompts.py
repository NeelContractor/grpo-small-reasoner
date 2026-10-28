"""Fixed prompting used by evaluation and training."""

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step.\n"
    "Respond exactly in this format:\n"
    "<think>\n"
    "...your reasoning...\n"
    "</think>\n"
    "<answer>\n"
    "...final numeric answer...\n"
    "</answer>"
)


def build_messages(question: str) -> list[dict[str, str]]:
    """Return the chat messages for one GSM8K question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {question.strip()}"},
    ]


def build_prompt(question: str, tokenizer) -> str:
    """Render the fixed chat prompt with the model's chat template."""
    return tokenizer.apply_chat_template(
        build_messages(question),
        tokenize=False,
        add_generation_prompt=True,
    )
