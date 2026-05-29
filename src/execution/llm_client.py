"""Provider-agnostic LLM client wrapper using xAI (Grok).

This module provides a thin wrapper around the xAI (Grok) API using the
OpenAI compatibility layer.
It exposes `send_messages(messages, **opts)` which accepts a list of chat
message dicts in the form {"role": "system|user|assistant", "content": "..."}
and returns the raw provider response object unchanged for auditability.

Environment variable used: `XAI_API_KEY` (expected by user to provide).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class GroqClient:
    """Wrapper for Groq API operations configured via environment variables."""

    def __init__(self) -> None:
        """Initialize the Groq client."""
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is missing. Please add it to your .env file.")
        
        # Groq provides an OpenAI-compatible endpoint
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def send_messages(self, messages: List[Dict[str, str]], **opts) -> Any:
        """Send a list of role-based messages to Groq via OpenAI compatible chat interface.

        Args:
            messages: List of dicts, e.g., [{"role": "user", "content": "Hello"}]
            opts: Additional kwargs like temperature or model.
        """
        model = opts.pop("model", "llama-3.3-70b-versatile")

        try:
            # Pass directly to OpenAI completion creation
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                **opts
            )
            return response
        except Exception as e:
            print(f"Grok API Error limit or failure: {e}")
            raise


def send_messages(messages: List[Dict[str, str]], **opts: Any) -> Any:
    """Convenience top-level function to send messages using env-configured client."""
    client = GroqClient()
    return client.send_messages(messages, **opts)
