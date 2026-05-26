from src.prompting import PromptRenderer, render_prompt


def test_renderer_lists_supported_catalog() -> None:
    renderer = PromptRenderer()

    assert renderer.list_tasks() == ("summarization", "code_generation", "reasoning")
    assert renderer.list_techniques() == (
        "zero_shot",
        "few_shot",
        "chain_of_thought",
        "tree_of_thought",
        "role_based",
        "structured_output",
    )


def test_zero_shot_render_builds_message_sections() -> None:
    prompt = render_prompt(
        task="summarization",
        technique="zero_shot",
        input_text="The quarterly results improved after the product launch.",
        summary_length="2 sentences",
        target_audience="executives",
    )

    assert prompt.system_prompt is not None
    assert "precise summarization assistant" in prompt.system_prompt
    assert "2 sentences" in prompt.user_prompt
    assert "executives" in prompt.user_prompt
    assert "quarterly results improved" in prompt.user_prompt
    assert prompt.as_messages()[0]["role"] == "system"
    assert prompt.as_messages()[1]["role"] == "user"


def test_few_shot_template_renders_examples() -> None:
    prompt = render_prompt(
        task="reasoning",
        technique="few_shot",
        problem_statement="If a car travels 60 miles in 2 hours, what is its average speed?",
        examples=[
            {
                "input": "If 2 pens cost 10 dollars, what does 1 pen cost?",
                "output": "Each pen costs 5 dollars.",
            }
        ],
        answer_format="a short sentence",
    )

    assert "Example 1:" in prompt.user_prompt
    assert "Each pen costs 5 dollars." in prompt.user_prompt
    assert "a short sentence" in prompt.user_prompt


def test_role_based_code_generation_uses_role_variable() -> None:
    prompt = render_prompt(
        task="code_generation",
        technique="role_based",
        problem_statement="Write a function that returns the maximum value in a list.",
        language="Python",
        role_name="a staff platform engineer",
    )

    assert prompt.system_prompt is not None
    assert "staff platform engineer" in prompt.system_prompt
    assert "Write clean, maintainable Python code." in prompt.user_prompt
