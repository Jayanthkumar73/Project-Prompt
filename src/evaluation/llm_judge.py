"""LLM-as-judge scoring pipeline using Gemini (google-generativeai).

Exposes `evaluate_with_llm(reference, generated)` which returns a dict with
accuracy, coherence, completeness, and overall scores. Returns safe defaults
on parse errors.
"""

import json
import os
from typing import Dict, Any

from ..execution.llm_client import GroqClient


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


def evaluate_with_llm(reference: str, generated: str, model: str = "llama-3.3-70b-versatile") -> Dict[str, Any]:
	"""Call Groq to evaluate and return parsed JSON scores.

	Falls back to zeroed scores on any failure.
	"""

	client = GroqClient()

	prompt = JUDGE_PROMPT.format(reference=reference, generated=generated)
	messages = [{"role": "user", "content": prompt}]

	try:
		# Use GroqClient to send messages
		response = client.send_messages(messages, model=model)
		
		# In openai structure, response text is here
		if hasattr(response, "choices") and response.choices:
			text = response.choices[0].message.content
		elif hasattr(response, "text"):
			text = response.text
		elif hasattr(response, "candidates") and response.candidates:
			# Older or different candidate format
			text = response.candidates[0].content.parts[0].text
		else:
			# fallback
			text = str(response)

		# Clean potential markdown code blocks if the model wrapped JSON
		text = text.strip()
		if text.startswith("```json"):
			text = text[7:]
		if text.endswith("```"):
			text = text[:-3]
		text = text.strip()

		model_client = gen.GenerativeModel(model_name=model)
		response = model_client.generate_content(prompt)
		text = getattr(response, "text", None) or response.output[0].content[0].text
		result = json.loads(text)
		return result
	except Exception as e:
		print(f"LLM Judge Error: {e}")
		return {
			"accuracy": 0,
			"coherence": 0,
			"completeness": 0,
			"overall": 0,
		}

