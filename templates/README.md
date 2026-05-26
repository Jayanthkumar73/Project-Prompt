Template layout: templates/{task}/{technique}.j2

Tasks:
- summarization
- code_generation
- reasoning

Techniques:
- zero_shot
- few_shot
- chain_of_thought
- tree_of_thought
- role_based
- structured_output

Authoring rules:
- Use `<<<SYSTEM>>>` and `<<<USER>>>` section markers.
- Keep task instructions inside the template, not in Python code.
- Prefer explicit variables over hidden defaults for experiment clarity.
