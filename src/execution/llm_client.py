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
	# New package name/namespace
	import google.genai as genai
except Exception:
	try:
		import google.generativeai as genai
	except Exception:  # pragma: no cover - defensive
		genai = None
try:
	from dotenv import load_dotenv
except Exception:
	load_dotenv = None


class GeminiClient:
	def __init__(self, api_key: str | None = None):
		# Attempt to load from .env if python-dotenv is available
		if load_dotenv is not None:
			# load .env from repo root if exists
			load_dotenv()

		key = api_key or os.getenv("GEMINI_API_KEY")
		if key is None:
			raise RuntimeError("GEMINI_API_KEY not set in environment")
		if genai is None:
			raise RuntimeError("google-generativeai package not installed")

		# For google.genai (v2+), we instantiate a Client.
		# For older google.generativeai, we use genai.configure()
		if hasattr(genai, "Client"):
			self._client = genai.Client(api_key=key)
		elif hasattr(genai, "configure"):
			genai.configure(api_key=key)
			self._client = genai
		else:
			# Fallback if neither works but package is imported
			self._client = genai

	def send_messages(self, messages: List[Dict[str, str]], **opts: Any) -> Any:
		"""Send chat messages to Gemini and return the raw provider response.

		This function maps the internal message format to Gemini's chat API.
		We keep the returned object unmodified to ensure raw persistence.
		"""
		# Mapping roles for google.genai: user, model
		# We'll use the 'contents' parameter for models.generate_content
		contents = []
		for m in messages:
			role = m.get("role")
			content = m.get("content")
			if role == "system":
				# Standard way is system_instruction in config, but let's just prepend to first user message if needed
				contents.append({"role": "user", "parts": [{"text": f"System Instruction: {content}"}]})
				continue
			
			gemini_role = "model" if role == "assistant" else "user"
			contents.append({"role": gemini_role, "parts": [{"text": content}]})

		# default model for normal generation runs
		model = opts.pop("model", "gemini-2.0-flash")

		# 1) Try client.models.generate_content (Best for one-shot/stateless calls)
		if hasattr(self._client, "models") and hasattr(self._client.models, "generate_content"):
			try:
				return self._client.models.generate_content(model=model, contents=contents, config=opts)
			except Exception as e:
				# Log the actual error for transparency
				if "429" in str(e):
					print(f"Gemini API Quota Exceeded (429) for model {model}.")
					raise
				
				# fallback to simple string if complex structure fails
				prompt_text = "\n".join([m.get("content", "") for m in messages])
				try:
					return self._client.models.generate_content(model=model, contents=prompt_text, config=opts)
				except Exception as e2:
					# if we get here, both attempts with 'model' failed
					# check if it's a 404 and we're not using 'latest'
					if "404" in str(e) and "latest" not in model:
						# try flash-latest as a ultimate fallback
						try:
							return self._client.models.generate_content(model="gemini-flash-latest", contents=contents, config=opts)
						except Exception:
							pass
					pass

		# 2) Fallback to chats if generate_content failed
		if hasattr(self._client, "chats") and hasattr(self._client.chats, "create"):
			try:
				# We send the last message as the actual message, others as history
				history = contents[:-1]
				last_msg = contents[-1]["parts"][0]["text"]
				chat = self._client.chats.create(model=model, history=history, config=opts)
				return chat.send_message(last_msg)
			except Exception as e:
				if "429" in str(e):
					raise
				pass

		# 3) Fallback for older SDK patterns
		if hasattr(self._client, "generate_text"):
			prompt_text = "\n".join([m.get("content", "") for m in messages])
			return self._client.generate_text(model=model, prompt=prompt_text, **opts)

		raise RuntimeError(f"Gemini API call failed for model {model}. Check console for specific error details.")


def send_messages(messages: List[Dict[str, str]], **opts: Any) -> Any:
	"""Convenience top-level function to send messages using env-configured client."""
	client = GeminiClient()
	return client.send_messages(messages, **opts)

