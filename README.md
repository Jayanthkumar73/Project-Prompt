# Prompt Lab

A reproducible prompt experimentation and evaluation platform.

## Scope
- Generate outputs for multiple prompting techniques
- Evaluate outputs with BLEU, ROUGE-L, BERTScore, and LLM-as-judge
- Measure variance and statistical significance
- Publish reproducible reports

## Frozen Team Ownership
- US-01 to US-07: Aditya
- US-08 to US-15: Samatham
- US-16 to US-25: Khushi

## Aditya Baseline
- Project bootstrap and local tooling
- Prompt registry and template renderer
- Prompting techniques for all three tasks

## Quick Start
1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Copy `.env.example` to `.env` and populate provider keys.
3. Use `src/prompting/prompt_renderer.py` to render prompts from `templates/`.
4. Run `pytest` to validate the prompting layer.

## Prompting Surface
- Supported tasks: `summarization`, `code_generation`, `reasoning`
- Supported techniques: `zero_shot`, `few_shot`, `chain_of_thought`, `tree_of_thought`, `role_based`, `structured_output`

See `docs/ARCHITECTURE.md`, `docs/OWNERSHIP.md`, `docs/WORKFLOW.md`, and `docs/PROMPTING.md`.
