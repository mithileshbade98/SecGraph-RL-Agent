"""
Pre-training data preparation utilities.

Load, validate, and preprocess datasets for RL training.

Supports:
- Loading from JSONL, CSV, Parquet
- Data validation and quality checks
- Formatting for PPO and DPO training
- Train/val/test splitting
- Data augmentation
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random
from loguru import logger
from collections import defaultdict
import numpy as np


@dataclass
class DatasetStatistics:
    """Statistics about a dataset."""

    num_examples: int
    avg_query_length: int
    avg_response_length: int
    unique_queries: int
    quality_score: float

    def __str__(self):
        return (
            f"Dataset Statistics:\n"
            f"  Examples: {self.num_examples}\n"
            f"  Avg Query Length: {self.avg_query_length}\n"
            f"  Avg Response Length: {self.avg_response_length}\n"
            f"  Unique Queries: {self.unique_queries}\n"
            f"  Quality Score: {self.quality_score:.2f}\n"
        )


class DatasetLoader:
    """
    Load datasets from various formats.

    Supports JSONL, CSV, Parquet, and custom formats.
    """

    @staticmethod
    def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
        """
        Load dataset from JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of examples
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        data = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line)
                    data.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num} due to JSON error: {e}")

        logger.info(f"Loaded {len(data)} examples from {file_path}")
        return data

    @staticmethod
    def load_csv(file_path: Path, delimiter: str = ',') -> List[Dict[str, Any]]:
        """
        Load dataset from CSV file.

        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter

        Returns:
            List of examples
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        data = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                data.append(dict(row))

        logger.info(f"Loaded {len(data)} examples from {file_path}")
        return data

    @staticmethod
    def load_parquet(file_path: Path) -> List[Dict[str, Any]]:
        """
        Load dataset from Parquet file.

        Args:
            file_path: Path to Parquet file

        Returns:
            List of examples
        """
        try:
            import pandas as pd

            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            df = pd.read_parquet(file_path)
            data = df.to_dict('records')

            logger.info(f"Loaded {len(data)} examples from {file_path}")
            return data

        except ImportError:
            logger.error("Parquet loading requires pandas. Install with: pip install pandas pyarrow")
            return []

    @staticmethod
    def save_jsonl(data: List[Dict[str, Any]], file_path: Path):
        """
        Save dataset to JSONL file.

        Args:
            data: List of examples
            file_path: Path to save JSONL file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            for example in data:
                f.write(json.dumps(example) + '\n')

        logger.info(f"Saved {len(data)} examples to {file_path}")


class DataValidator:
    """
    Validate dataset quality and format.
    """

    @staticmethod
    def validate_ppo_format(data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate that data is in correct format for PPO training.

        Required fields:
        - query: str
        - response: str
        - reward: float

        Args:
            data: List of examples

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        required_fields = ['query', 'response', 'reward']

        for i, example in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in example:
                    errors.append(f"Example {i}: Missing required field '{field}'")

            # Validate types
            if 'query' in example and not isinstance(example['query'], str):
                errors.append(f"Example {i}: 'query' must be a string")

            if 'response' in example and not isinstance(example['response'], str):
                errors.append(f"Example {i}: 'response' must be a string")

            if 'reward' in example:
                try:
                    float(example['reward'])
                except (ValueError, TypeError):
                    errors.append(f"Example {i}: 'reward' must be a number")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def validate_dpo_format(data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate that data is in correct format for DPO training.

        Required fields:
        - prompt: str
        - chosen: str
        - rejected: str

        Args:
            data: List of examples

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        required_fields = ['prompt', 'chosen', 'rejected']

        for i, example in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in example:
                    errors.append(f"Example {i}: Missing required field '{field}'")

            # Validate types
            for field in required_fields:
                if field in example and not isinstance(example[field], str):
                    errors.append(f"Example {i}: '{field}' must be a string")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def compute_statistics(data: List[Dict[str, Any]]) -> DatasetStatistics:
        """
        Compute statistics about the dataset.

        Args:
            data: List of examples

        Returns:
            DatasetStatistics object
        """
        num_examples = len(data)

        # Compute lengths
        query_lengths = []
        response_lengths = []
        unique_queries = set()

        for example in data:
            # Handle different field names
            query = example.get('query', example.get('prompt', ''))
            response = example.get('response', example.get('chosen', ''))

            query_lengths.append(len(query.split()))
            response_lengths.append(len(response.split()))
            unique_queries.add(query)

        avg_query_length = int(np.mean(query_lengths)) if query_lengths else 0
        avg_response_length = int(np.mean(response_lengths)) if response_lengths else 0

        # Compute quality score (simple heuristic)
        quality_score = 1.0
        if avg_query_length < 5:
            quality_score -= 0.2
        if avg_response_length < 10:
            quality_score -= 0.2
        if len(unique_queries) < num_examples * 0.5:
            quality_score -= 0.3  # Many duplicate queries

        quality_score = max(0.0, min(1.0, quality_score))

        return DatasetStatistics(
            num_examples=num_examples,
            avg_query_length=avg_query_length,
            avg_response_length=avg_response_length,
            unique_queries=len(unique_queries),
            quality_score=quality_score,
        )


class DataPreprocessor:
    """
    Preprocess and format datasets for training.
    """

    def __init__(self, tokenizer=None):
        """
        Initialize preprocessor.

        Args:
            tokenizer: Optional tokenizer for length validation
        """
        self.tokenizer = tokenizer

    def format_for_ppo(
        self,
        data: List[Dict[str, Any]],
        query_field: str = 'query',
        response_field: str = 'response',
        reward_field: str = 'reward',
    ) -> List[Dict[str, Any]]:
        """
        Format data for PPO training.

        Args:
            data: Raw data
            query_field: Name of query field
            response_field: Name of response field
            reward_field: Name of reward field

        Returns:
            Formatted data
        """
        formatted = []

        for example in data:
            if query_field in example and response_field in example:
                formatted_example = {
                    'query': example[query_field],
                    'response': example[response_field],
                    'reward': example.get(reward_field, 0.0),
                    'value': example.get('value', [0.0]),  # Will be computed during training
                    'advantage': example.get('advantage', [0.0]),  # Will be computed during training
                    'old_log_prob': example.get('old_log_prob', [0.0]),  # Will be computed during training
                }
                formatted.append(formatted_example)

        logger.info(f"Formatted {len(formatted)} examples for PPO")
        return formatted

    def format_for_dpo(
        self,
        data: List[Dict[str, Any]],
        prompt_field: str = 'prompt',
        chosen_field: str = 'chosen',
        rejected_field: str = 'rejected',
    ) -> List[Dict[str, Any]]:
        """
        Format data for DPO training.

        Args:
            data: Raw data
            prompt_field: Name of prompt field
            chosen_field: Name of chosen completion field
            rejected_field: Name of rejected completion field

        Returns:
            Formatted data
        """
        formatted = []

        for example in data:
            if all(f in example for f in [prompt_field, chosen_field, rejected_field]):
                formatted_example = {
                    'prompt': example[prompt_field],
                    'chosen': example[chosen_field],
                    'rejected': example[rejected_field],
                    'confidence': example.get('confidence', 1.0),
                }
                formatted.append(formatted_example)

        logger.info(f"Formatted {len(formatted)} examples for DPO")
        return formatted

    def filter_by_length(
        self,
        data: List[Dict[str, Any]],
        max_query_length: int = 512,
        max_response_length: int = 512,
        min_query_length: int = 5,
        min_response_length: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Filter examples by length constraints.

        Args:
            data: List of examples
            max_query_length: Maximum query length (in tokens if tokenizer provided, else words)
            max_response_length: Maximum response length
            min_query_length: Minimum query length
            min_response_length: Minimum response length

        Returns:
            Filtered data
        """
        filtered = []

        for example in data:
            query = example.get('query', example.get('prompt', ''))
            response = example.get('response', example.get('chosen', ''))

            # Compute lengths
            if self.tokenizer:
                query_len = len(self.tokenizer.encode(query))
                response_len = len(self.tokenizer.encode(response))
            else:
                query_len = len(query.split())
                response_len = len(response.split())

            # Check constraints
            if (min_query_length <= query_len <= max_query_length and
                min_response_length <= response_len <= max_response_length):
                filtered.append(example)

        logger.info(f"Filtered dataset: {len(data)} -> {len(filtered)} examples")
        return filtered

    def deduplicate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate examples based on query.

        Args:
            data: List of examples

        Returns:
            Deduplicated data
        """
        seen_queries = set()
        deduplicated = []

        for example in data:
            query = example.get('query', example.get('prompt', ''))

            if query not in seen_queries:
                seen_queries.add(query)
                deduplicated.append(example)

        logger.info(f"Deduplicated dataset: {len(data)} -> {len(deduplicated)} examples")
        return deduplicated

    def balance_dataset(
        self,
        data: List[Dict[str, Any]],
        category_field: str = 'category',
        max_per_category: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Balance dataset across categories.

        Args:
            data: List of examples
            category_field: Field containing category labels
            max_per_category: Maximum examples per category

        Returns:
            Balanced data
        """
        # Group by category
        categories = defaultdict(list)
        for example in data:
            category = example.get(category_field, 'default')
            categories[category].append(example)

        # Determine max per category
        if max_per_category is None:
            max_per_category = min(len(examples) for examples in categories.values())

        # Sample from each category
        balanced = []
        for category, examples in categories.items():
            sampled = random.sample(examples, min(len(examples), max_per_category))
            balanced.extend(sampled)

        logger.info(f"Balanced dataset: {len(data)} -> {len(balanced)} examples across {len(categories)} categories")
        return balanced


class DataSplitter:
    """
    Split datasets into train/validation/test sets.
    """

    @staticmethod
    def train_val_test_split(
        data: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split data into train/val/test sets.

        Args:
            data: List of examples
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            shuffle: Whether to shuffle before splitting
            seed: Random seed

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        # Shuffle if requested
        if shuffle:
            random.seed(seed)
            data = data.copy()
            random.shuffle(data)

        # Compute split indices
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        logger.info(f"Split dataset: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        return train_data, val_data, test_data

    @staticmethod
    def k_fold_split(
        data: List[Dict[str, Any]],
        k: int = 5,
        shuffle: bool = True,
        seed: int = 42,
    ) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """
        Create k-fold cross-validation splits.

        Args:
            data: List of examples
            k: Number of folds
            shuffle: Whether to shuffle before splitting
            seed: Random seed

        Returns:
            List of (train, val) tuples for each fold
        """
        # Shuffle if requested
        if shuffle:
            random.seed(seed)
            data = data.copy()
            random.shuffle(data)

        # Compute fold size
        n = len(data)
        fold_size = n // k

        folds = []
        for i in range(k):
            # Validation set for this fold
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < k - 1 else n
            val_data = data[val_start:val_end]

            # Training set is everything else
            train_data = data[:val_start] + data[val_end:]

            folds.append((train_data, val_data))

        logger.info(f"Created {k}-fold splits")
        return folds


def prepare_reasoning_dataset(
    input_path: Path,
    output_dir: Path,
    format_type: str = 'ppo',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, Path]:
    """
    Convenience function to prepare a reasoning dataset.

    Args:
        input_path: Path to input data file (JSONL)
        output_dir: Directory to save processed splits
        format_type: 'ppo' or 'dpo'
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Dictionary with paths to train/val/test files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {input_path}")
    loader = DataLoader()
    data = loader.load_jsonl(input_path)

    # Validate
    validator = DataValidator()
    if format_type == 'ppo':
        is_valid, errors = validator.validate_ppo_format(data)
    else:
        is_valid, errors = validator.validate_dpo_format(data)

    if not is_valid:
        logger.warning(f"Data validation found {len(errors)} errors")
        for error in errors[:10]:  # Show first 10
            logger.warning(error)

    # Compute statistics
    stats = validator.compute_statistics(data)
    logger.info(str(stats))

    # Preprocess
    preprocessor = DataPreprocessor()
    data = preprocessor.filter_by_length(data)
    data = preprocessor.deduplicate(data)

    # Format
    if format_type == 'ppo':
        data = preprocessor.format_for_ppo(data)
    else:
        data = preprocessor.format_for_dpo(data)

    # Split
    splitter = DataSplitter()
    train_data, val_data, test_data = splitter.train_val_test_split(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # Save splits
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    loader.save_jsonl(train_data, train_path)
    loader.save_jsonl(val_data, val_path)
    loader.save_jsonl(test_data, test_path)

    logger.success(f"Dataset prepared in {output_dir}")

    return {
        'train': train_path,
        'val': val_path,
        'test': test_path,
    }


# Example usage
if __name__ == "__main__":
    # Create mock dataset
    mock_data = [
        {
            'query': 'Detect multi-account abuse',
            'response': 'Step 1: Check device fingerprints...',
            'reward': 0.8,
        },
        {
            'query': 'Find payment fraud patterns',
            'response': 'Step 1: Analyze transaction history...',
            'reward': 0.9,
        },
    ] * 50  # 100 examples

    # Save mock data
    output_dir = Path("data/processed/reasoning")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DatasetLoader()
    loader.save_jsonl(mock_data, output_dir / "raw.jsonl")

    # Prepare dataset
    paths = prepare_reasoning_dataset(
        input_path=output_dir / "raw.jsonl",
        output_dir=output_dir / "splits",
        format_type='ppo',
    )

    print(f"Prepared dataset:")
    print(f"  Train: {paths['train']}")
    print(f"  Val: {paths['val']}")
    print(f"  Test: {paths['test']}")
