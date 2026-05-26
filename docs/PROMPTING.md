# Prompting Layer

This document freezes the Aditya-owned prompting surface for `US-01` to `US-07`.

## Techniques Implemented
- `zero_shot`
- `few_shot`
- `chain_of_thought`
- `tree_of_thought`
- `role_based`
- `structured_output`

## Template Contract
- Each template lives at `templates/{task}/{technique}.j2`.
- Templates may define `<<<SYSTEM>>>`, `<<<USER>>>`, and optional `<<<ASSISTANT>>>`.
- If no explicit section markers are present, the full template is treated as the user message.

## Renderer API
- `PromptRenderer.render(task, technique, **variables)` returns a `RenderedPrompt`.
- `RenderedPrompt.as_messages()` converts the rendered prompt into chat-style messages.

## Ownership Boundary
- Aditya owns prompt authoring, technique framing, and template rendering.
- Samatham should consume rendered prompts from `src/prompting/` rather than duplicating prompt logic in the runner.
