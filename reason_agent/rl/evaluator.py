"""
Validation and evaluation utilities for trained models.

Evaluate models on held-out datasets to measure performance.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from tqdm import tqdm

from .lora_utils import generate_with_lora, preprocess_batch, postprocess_output


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    loss: float
    accuracy: float
    perplexity: float
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None
    exact_match: Optional[float] = None
    f1_score: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        result = {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'perplexity': self.perplexity,
        }

        if self.bleu_score is not None:
            result['bleu_score'] = self.bleu_score
        if self.rouge_scores is not None:
            result.update({f'rouge_{k}': v for k, v in self.rouge_scores.items()})
        if self.exact_match is not None:
            result['exact_match'] = self.exact_match
        if self.f1_score is not None:
            result['f1_score'] = self.f1_score

        return result


class ModelEvaluator:
    """
    Evaluates trained models on validation/test sets.

    Supports various metrics:
    - Loss and perplexity
    - Accuracy (exact match)
    - BLEU score (text generation quality)
    - ROUGE scores (summarization quality)
    - Task-specific metrics
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate (can be PEFT model)
            tokenizer: Tokenizer
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        # Move model to device
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)

        logger.info(f"Evaluator initialized on {self.device}")

    def evaluate(
        self,
        eval_data: List[Dict[str, Any]],
        metrics: List[str] = ['loss', 'accuracy', 'perplexity'],
        max_length: int = 512,
        show_progress: bool = True,
    ) -> EvaluationMetrics:
        """
        Evaluate model on dataset.

        Args:
            eval_data: List of examples with 'input' and 'target' keys
            metrics: List of metrics to compute
            max_length: Maximum sequence length
            show_progress: Whether to show progress bar

        Returns:
            EvaluationMetrics object
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_examples = len(eval_data)

        predictions = []
        references = []

        # Process in batches
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size

        iterator = range(0, num_examples, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Evaluating", total=num_batches)

        with torch.no_grad():
            for batch_start in iterator:
                batch_end = min(batch_start + self.batch_size, num_examples)
                batch = eval_data[batch_start:batch_end]

                # Extract inputs and targets
                inputs = [ex['input'] for ex in batch]
                targets = [ex['target'] for ex in batch]

                # Tokenize inputs
                input_encodings = self.tokenizer(
                    inputs,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )

                # Tokenize targets (for loss computation)
                target_encodings = self.tokenizer(
                    targets,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )

                # Move to device
                input_ids = input_encodings['input_ids'].to(self.device)
                attention_mask = input_encodings['attention_mask'].to(self.device)
                target_ids = target_encodings['input_ids'].to(self.device)

                # Compute loss if needed
                if 'loss' in metrics or 'perplexity' in metrics:
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=target_ids,
                    )

                    batch_loss = outputs.loss.item() if hasattr(outputs, 'loss') else 0.0
                    total_loss += batch_loss * len(batch)

                # Generate predictions if needed
                if any(m in metrics for m in ['accuracy', 'bleu', 'rouge', 'exact_match', 'f1']):
                    # Generate
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=False,  # Greedy decoding for evaluation
                    )

                    # Decode predictions
                    for i, gen_ids in enumerate(generated_ids):
                        # Skip input tokens
                        input_len = input_ids[i].shape[0]
                        pred_ids = gen_ids[input_len:]

                        pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                        predictions.append(pred_text)
                        references.append(targets[i])

                        # Compute accuracy (exact match)
                        if 'accuracy' in metrics or 'exact_match' in metrics:
                            if pred_text.strip() == targets[i].strip():
                                total_correct += 1

                # Count tokens for perplexity
                total_tokens += attention_mask.sum().item()

        # Compute final metrics
        avg_loss = total_loss / num_examples if num_examples > 0 else 0.0
        accuracy = total_correct / num_examples if num_examples > 0 else 0.0
        perplexity = np.exp(avg_loss) if avg_loss > 0 else float('inf')

        # Compute additional metrics
        bleu_score = None
        rouge_scores = None
        f1_score = None

        if predictions and references:
            if 'bleu' in metrics:
                bleu_score = self._compute_bleu(predictions, references)

            if 'rouge' in metrics:
                rouge_scores = self._compute_rouge(predictions, references)

            if 'f1' in metrics:
                f1_score = self._compute_f1(predictions, references)

        logger.info(f"Evaluation complete: loss={avg_loss:.4f}, accuracy={accuracy:.4f}, perplexity={perplexity:.2f}")

        return EvaluationMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            perplexity=perplexity,
            bleu_score=bleu_score,
            rouge_scores=rouge_scores,
            exact_match=accuracy,  # Same as accuracy for now
            f1_score=f1_score,
        )

    def _compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute BLEU score.

        Simple implementation - for production use sacrebleu library.
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu

            scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]

                if len(pred_tokens) > 0 and len(ref_tokens[0]) > 0:
                    score = sentence_bleu(ref_tokens, pred_tokens)
                    scores.append(score)

            return np.mean(scores) if scores else 0.0
        except ImportError:
            logger.warning("NLTK not available, skipping BLEU computation")
            return 0.0

    def _compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores.

        Simple implementation - for production use rouge-score library.
        """
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []

            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)

            return {
                'rouge1': np.mean(rouge1_scores),
                'rouge2': np.mean(rouge2_scores),
                'rougeL': np.mean(rougeL_scores),
            }
        except ImportError:
            logger.warning("rouge-score not available, skipping ROUGE computation")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def _compute_f1(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute token-level F1 score.
        """
        f1_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())

            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1_scores.append(0.0)
                continue

            # Compute precision, recall, F1
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            recall = len(common) / len(ref_tokens) if len(ref_tokens) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            f1_scores.append(f1)

        return np.mean(f1_scores) if f1_scores else 0.0

    def evaluate_with_generation(
        self,
        eval_data: List[Dict[str, Any]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model using generation (for reasoning tasks).

        Args:
            eval_data: List of examples with 'query' and 'expected_output' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with metrics and generated outputs
        """
        self.model.eval()

        outputs = []
        correct = 0

        iterator = eval_data
        if show_progress:
            iterator = tqdm(eval_data, desc="Generating")

        for example in iterator:
            query = example['query']
            expected = example.get('expected_output', '')

            # Generate response
            try:
                response = generate_with_lora(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=query,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )

                # Check correctness (exact match or custom evaluator)
                is_correct = response.strip().lower() == expected.strip().lower()
                if is_correct:
                    correct += 1

                outputs.append({
                    'query': query,
                    'expected': expected,
                    'generated': response,
                    'is_correct': is_correct,
                })
            except Exception as e:
                logger.error(f"Generation failed for query '{query}': {e}")
                outputs.append({
                    'query': query,
                    'expected': expected,
                    'generated': '',
                    'is_correct': False,
                })

        accuracy = correct / len(eval_data) if len(eval_data) > 0 else 0.0

        logger.info(f"Generation evaluation complete: accuracy={accuracy:.4f}")

        return {
            'accuracy': accuracy,
            'num_correct': correct,
            'num_total': len(eval_data),
            'outputs': outputs,
        }


def evaluate_on_dataset(
    model,
    tokenizer,
    dataset_path: Path,
    metrics: List[str] = ['loss', 'accuracy'],
    batch_size: int = 8,
) -> EvaluationMetrics:
    """
    Convenience function to evaluate on a dataset file.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset_path: Path to dataset (JSONL format)
        metrics: Metrics to compute
        batch_size: Batch size

    Returns:
        EvaluationMetrics object
    """
    import json

    # Load dataset
    eval_data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            eval_data.append(example)

    logger.info(f"Loaded {len(eval_data)} examples from {dataset_path}")

    # Create evaluator and evaluate
    evaluator = ModelEvaluator(model, tokenizer, batch_size=batch_size)
    return evaluator.evaluate(eval_data, metrics=metrics)


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Mock evaluation data
    eval_data = [
        {'input': 'What is 2+2?', 'target': '4'},
        {'input': 'What is the capital of France?', 'target': 'Paris'},
        {'input': 'What is 10*5?', 'target': '50'},
    ]

    # Load model
    model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

    # Evaluate
    evaluator = ModelEvaluator(model, tokenizer)
    metrics = evaluator.evaluate(
        eval_data,
        metrics=['loss', 'accuracy', 'perplexity'],
    )

    print(f"Metrics: {metrics.to_dict()}")
