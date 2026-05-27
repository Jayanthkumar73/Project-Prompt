"""Experiment orchestration and raw response persistence.

This runner loads a dataset JSON file (array of cases), renders prompts via the
prompting layer, calls the LLM client, evaluates metrics, and persists per-run
records to `results/raw_responses.jsonl` and an aggregated CSV to
`results/scores.csv`.
"""

from __future__ import annotations

import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .storage import append_jsonl, read_jsonl
from ..prompting.prompt_renderer import render_prompt
from .llm_client import send_messages
from ..evaluation.metrics import evaluate_response
from ..evaluation.llm_judge import evaluate_with_llm

import pandas as pd


RESULTS_RAW = Path("results/raw_responses.jsonl")
RESULTS_SCORES = Path("results/scores.csv")


def run_experiment(dataset_path: Path | str, task: str, technique: str, repeats: int = 3) -> None:
	dataset_path = Path(dataset_path)
	with dataset_path.open("r", encoding="utf-8") as fh:
		cases = json.load(fh)

	rows: list[dict[str, Any]] = []

	for case in cases:
		case_id = case.get("id")
		reference = case.get("reference")
		input_payload = case.get("input")

		for run_index in range(1, repeats + 1):
			print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing Task: {task} | Technique: {technique} | Case: {case_id} | Repeat: {run_index}/{repeats}...")
			run_id = str(uuid.uuid4())
			ts = datetime.utcnow().isoformat() + "Z"

			# Render prompt using prompt_renderer
			rendered = render_prompt(task=task, technique=technique, input_text=input_payload)
			messages = rendered.as_messages()

			# Call provider and get raw response (kept unchanged)
			raw_response = send_messages(messages)

			# Extract generated text conservatively
			try:
				if hasattr(raw_response, "text"):
					generated = raw_response.text
				elif hasattr(raw_response, "candidates") and raw_response.candidates:
					generated = raw_response.candidates[0].content.parts[0].text
				else:
					generated = str(raw_response)
			except Exception:
				# Fallback: stringify the response
				generated = str(raw_response)

			# Evaluate metrics
			metrics = evaluate_response(reference=reference, generated=generated)
			judge = evaluate_with_llm(reference=reference, generated=generated)

			# Free tier rate limiting: 1 request every 12 seconds to be safe
			# Each repeat makes 2 LLM calls (Generation + Judge)
			time.sleep(12)

			record = {
				"run_id": run_id,
				"timestamp": ts,
				"case_id": case_id,
				"task": task,
				"technique": technique,
				"run_index": run_index,
				"rendered_prompt": rendered.raw_text,
				"messages": messages,
				"raw_response": str(raw_response),
				"generated": generated,
				"metrics": metrics,
				"judge": judge,
			}

			append_jsonl(RESULTS_RAW, record)

			rows.append(
				{
					"case_id": case_id,
					"technique": technique,
					"run_index": run_index,
					"bleu": metrics["bleu"],
					"rouge_l": metrics["rouge_l"],
					"bertscore": metrics["bertscore"],
					"judge_accuracy": judge.get("accuracy"),
					"judge_coherence": judge.get("coherence"),
					"judge_completeness": judge.get("completeness"),
					"judge_overall": judge.get("overall"),
				}
			)

	# Persist aggregated per-run CSV
	df = pd.DataFrame(rows)
	df.to_csv(RESULTS_SCORES, index=False)


def run_all_techniques(dataset_path: Path | str, task: str, repeats: int = 3) -> None:
	"""Runs all supported prompt techniques for a given task and dataset."""
	from src.prompting.prompt_catalog import SUPPORTED_TECHNIQUES
	
	for technique in SUPPORTED_TECHNIQUES:
		print(f"\n{'='*50}\nStarting testing for technique: {technique.upper()}\n{'='*50}")
		run_experiment(dataset_path, task, technique, repeats)

def load_raw_results(path: Path | str = RESULTS_RAW):
	return list(read_jsonl(Path(path)))


if __name__ == "__main__":
	# Example invocation: run against summarization test cases with 3 repeats
	run_experiment("data/summarization/test_cases.json", task="summarization", technique="zero_shot", repeats=3)

