"""Canonical prompt task and technique catalog."""

SUPPORTED_TASKS: tuple[str, ...] = (
    "summarization",
    "code_generation",
    "reasoning",
)

SUPPORTED_TECHNIQUES: tuple[str, ...] = (
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "tree_of_thought",
    "role_based",
    "structured_output",
)
