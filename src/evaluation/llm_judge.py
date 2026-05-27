"""LLM-as-judge scoring pipeline using Gemini (google-generativeai).

Exposes `evaluate_with_llm(reference, generated)` which returns a dict with
accuracy, coherence, completeness, and overall scores. Returns safe defaults
on parse errors.
"""

import json
import os
from typing import Dict, Any

import google.generativeai as genai


JUDGE_PROMPT = """
You are an expert evaluator.

Task:
Evaluate the quality of the generated response.

Reference Answer:
{reference}

Generated Response:
{generated}

Evaluate on:
1. Accuracy (1-10)
2. Coherence (1-10)
3. Completeness (1-10)

Return ONLY valid JSON:

{{
	"accuracy": number,
	"coherence": number,
	"completeness": number,
	"overall": number
}}
"""


def _configure_genai():
	key = os.getenv("GEMINI_API_KEY")
	if not key:
		raise RuntimeError("GEMINI_API_KEY not set in environment")
	genai.configure(api_key=key)
	return genai


def evaluate_with_llm(reference: str, generated: str, model: str = "gemini-3.1-pro") -> Dict[str, Any]:
	"""Call Gemini to evaluate and return parsed JSON scores.

	Falls back to zeroed scores on any failure.
	"""

	gen = _configure_genai()

	prompt = JUDGE_PROMPT.format(reference=reference, generated=generated)

	try:
		model_client = gen.GenerativeModel(model_name=model)
		response = model_client.generate_content(prompt)
		text = getattr(response, "text", None) or response.output[0].content[0].text
		result = json.loads(text)
		return result
	except Exception:
		return {
			"accuracy": 0,
			"coherence": 0,
			"completeness": 0,
			"overall": 0,
		}

