"""Aggregate per-run results and compute variance statistics.

Reads `results/raw_responses.jsonl` and outputs `results/summary.csv` and
`results/summary.json` with mean, std, and 95% CI for BLEU, ROUGE-L, and BERTScore
grouped by task and technique.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .variance import mean_std_ci
from ..execution.storage import read_jsonl

RESULTS_RAW = Path("results/raw_responses.jsonl")
SUMMARY_CSV = Path("results/summary.csv")
SUMMARY_JSON = Path("results/summary.json")


def aggregate_results(raw_path: Path | str = RESULTS_RAW) -> Dict[str, Dict]:
    records = list(read_jsonl(Path(raw_path)))
    # group by (task, technique)
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in records:
        key = (r.get("task"), r.get("technique"))
        groups[key].append(r)

    summary: Dict[str, Dict] = {}
    for (task, technique), items in groups.items():
        bleus = [it["metrics"]["bleu"] for it in items if it.get("metrics")]
        rouges = [it["metrics"]["rouge_l"] for it in items if it.get("metrics")]
        berts = [it["metrics"]["bertscore"] for it in items if it.get("metrics")]

        bleu_stats = mean_std_ci(bleus)
        rouge_stats = mean_std_ci(rouges)
        bert_stats = mean_std_ci(berts)

        key_name = f"{task}::{technique}"
        summary[key_name] = {
            "task": task,
            "technique": technique,
            "n_runs": len(items),
            "bleu": bleu_stats,
            "rouge_l": rouge_stats,
            "bertscore": bert_stats,
        }

    # write outputs
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # build CSV rows
    import csv

    with SUMMARY_CSV.open("w", newline='', encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "task",
            "technique",
            "n_runs",
            "bleu_mean",
            "bleu_std",
            "bleu_ci_low",
            "bleu_ci_high",
            "rouge_mean",
            "rouge_std",
            "rouge_ci_low",
            "rouge_ci_high",
            "bertscore_mean",
            "bertscore_std",
            "bertscore_ci_low",
            "bertscore_ci_high",
        ])
        for v in summary.values():
            b = v["bleu"]
            r = v["rouge_l"]
            br = v["bertscore"]
            writer.writerow([
                v["task"],
                v["technique"],
                v["n_runs"],
                b["mean"],
                b["std"],
                b["ci_low"],
                b["ci_high"],
                r["mean"],
                r["std"],
                r["ci_low"],
                r["ci_high"],
                br["mean"],
                br["std"],
                br["ci_low"],
                br["ci_high"],
            ])

    return summary


if __name__ == "__main__":
    s = aggregate_results()
    print("Wrote summary for", len(s), "groups")
