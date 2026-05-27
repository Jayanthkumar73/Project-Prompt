"""Provider-agnostic LLM client wrapper.

This module provides a thin wrapper around Google Gemini (Generative AI) SDK.
It exposes `send_messages(messages, **opts)` which accepts a list of chat
message dicts in the form {"role": "system|user|assistant", "content": "..."}
and returns the raw provider response object unchanged for auditability.

Environment variable used: `GEMINI_API_KEY` (expected by user to provide).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

try:
	import google.generativeai as genai
except Exception:  # pragma: no cover - defensive
	genai = None


class GeminiClient:
	def __init__(self, api_key: str | None = None):
		key = api_key or os.getenv("GEMINI_API_KEY")
		if key is None:
			raise RuntimeError("GEMINI_API_KEY not set in environment")
		if genai is None:
			raise RuntimeError("google-generativeai package not installed")
		genai.configure(api_key=key)
		self._client = genai

	def send_messages(self, messages: List[Dict[str, str]], **opts: Any) -> Any:
		"""Send chat messages to Gemini and return the raw provider response.

		This function maps the internal message format to Gemini's chat API.
		We keep the returned object unmodified to ensure raw persistence.
		"""
		# Map roles to Gemini expected structure: list of dicts with 'author' and 'content'
		chat_messages = []
		for m in messages:
			role = m.get("role")
			content = m.get("content")
			if role == "system":
				author = "system"
			elif role == "assistant":
				author = "assistant"
			else:
				author = "user"
			chat_messages.append({"author": author, "content": content})

		# Call the Gemeni chat completions API (synchronous)
		# Users can pass model and other kwargs via opts (e.g., model='gemini-pro')
		# default model for normal generation runs
		model = opts.pop("model", "gemini-2.5-flash")
		response = self._client.chat.create(model=model, messages=chat_messages, **opts)
		return response


def send_messages(messages: List[Dict[str, str]], **opts: Any) -> Any:
	"""Convenience top-level function to send messages using env-configured client."""
	client = GeminiClient()
	return client.send_messages(messages, **opts)

