"""BLEU, ROUGE-L, and BERTScore evaluation functions.

Provides:
- compute_bleu
- compute_rouge_l
- compute_bertscore
- evaluate_response (master wrapper)

All functions return rounded float scores as requested.
"""

from typing import Dict

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

from rouge_score import rouge_scorer

from bert_score import score as bert_score


def compute_bleu(reference: str, generated: str) -> float:
	"""
	Compute BLEU score between reference and generated text.
	"""

	reference_tokens = [word_tokenize(reference.lower())]
	generated_tokens = word_tokenize(generated.lower())

	bleu_score = sentence_bleu(
		reference_tokens,
		generated_tokens
	)

	return round(bleu_score, 4)


def compute_rouge_l(reference: str, generated: str) -> float:
	"""
	Compute ROUGE-L F1 score.
	"""

	scorer = rouge_scorer.RougeScorer(
		["rougeL"],
		use_stemmer=True
	)

	scores = scorer.score(reference, generated)

	rouge_l = scores["rougeL"].fmeasure

	return round(rouge_l, 4)


def compute_bertscore(reference: str, generated: str) -> float:
	"""
	Compute BERTScore F1.
	"""

	P, R, F1 = bert_score(
		[generated],
		[reference],
		lang="en",
		model_type="distilbert-base-uncased"
	)

	return round(F1.mean().item(), 4)


def evaluate_response(reference: str, generated: str) -> Dict[str, float]:
	"""
	Run all evaluation metrics and return a dict of scores.
	"""

	results = {
		"bleu": compute_bleu(reference, generated),
		"rouge_l": compute_rouge_l(reference, generated),
		"bertscore": compute_bertscore(reference, generated),
	}

	return results

