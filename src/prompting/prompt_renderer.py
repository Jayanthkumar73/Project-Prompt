"""Prompt template loading and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from .prompt_catalog import SUPPORTED_TASKS, SUPPORTED_TECHNIQUES

SECTION_MARKER_PATTERN = re.compile(r"^<<<(?P<section>SYSTEM|USER|ASSISTANT)>>>$", re.MULTILINE)


class PromptTemplateError(ValueError):
    """Base exception for template rendering and parsing failures."""


class UnsupportedTaskError(PromptTemplateError):
    """Raised when the requested task is not part of the prompt catalog."""


class UnsupportedTechniqueError(PromptTemplateError):
    """Raised when the requested technique is not part of the prompt catalog."""


class TemplateSectionError(PromptTemplateError):
    """Raised when a template cannot be parsed into supported prompt sections."""


@dataclass(frozen=True)
class RenderedPrompt:
    """Rendered prompt payload prepared for downstream LLM clients."""

    task: str
    technique: str
    system_prompt: str | None
    user_prompt: str
    assistant_prompt: str | None
    raw_text: str
    template_path: Path
    variables: dict[str, Any]

    def as_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.user_prompt})
        if self.assistant_prompt:
            messages.append({"role": "assistant", "content": self.assistant_prompt})
        return messages


class PromptRenderer:
    """Loads Jinja2 templates and returns structured prompt sections."""

    def __init__(self, template_root: Path | None = None) -> None:
        self.template_root = template_root or Path(__file__).resolve().parents[2] / "templates"
        self.environment = Environment(
            loader=FileSystemLoader(str(self.template_root)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
            undefined=StrictUndefined,
        )

    def list_tasks(self) -> tuple[str, ...]:
        return SUPPORTED_TASKS

    def list_techniques(self) -> tuple[str, ...]:
        return SUPPORTED_TECHNIQUES

    def render(self, task: str, technique: str, **variables: Any) -> RenderedPrompt:
        self._validate_task(task)
        self._validate_technique(technique)

        template_name = f"{task}/{technique}.j2"
        try:
            template = self.environment.get_template(template_name)
        except TemplateNotFound as exc:
            raise PromptTemplateError(
                f"Template '{template_name}' was not found under '{self.template_root}'."
            ) from exc

        raw_text = template.render(**variables).strip()
        sections = self._parse_sections(raw_text)
        template_path = self.template_root / template_name

        return RenderedPrompt(
            task=task,
            technique=technique,
            system_prompt=sections.get("SYSTEM"),
            user_prompt=sections["USER"],
            assistant_prompt=sections.get("ASSISTANT"),
            raw_text=raw_text,
            template_path=template_path,
            variables=dict(variables),
        )

    def _validate_task(self, task: str) -> None:
        if task not in SUPPORTED_TASKS:
            supported = ", ".join(SUPPORTED_TASKS)
            raise UnsupportedTaskError(f"Unsupported task '{task}'. Supported tasks: {supported}.")

    def _validate_technique(self, technique: str) -> None:
        if technique not in SUPPORTED_TECHNIQUES:
            supported = ", ".join(SUPPORTED_TECHNIQUES)
            raise UnsupportedTechniqueError(
                f"Unsupported technique '{technique}'. Supported techniques: {supported}."
            )

    def _parse_sections(self, raw_text: str) -> dict[str, str]:
        matches = list(SECTION_MARKER_PATTERN.finditer(raw_text))
        if not matches:
            return {"USER": raw_text.strip()}

        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            section_name = match.group("section")
            content_start = match.end()
            content_end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
            content = raw_text[content_start:content_end].strip()
            if not content:
                raise TemplateSectionError(f"Section '{section_name}' is empty.")
            sections[section_name] = content

        if "USER" not in sections:
            raise TemplateSectionError("Template must define a USER section.")

        return sections


def render_prompt(task: str, technique: str, **variables: Any) -> RenderedPrompt:
    """Convenience wrapper for single-shot rendering."""

    renderer = PromptRenderer()
    return renderer.render(task=task, technique=technique, **variables)
