"""
Data collator utilities for RL training.

Provides collators for:
- PPO rollout batching
- DPO preference pair batching
- Dynamic padding and truncation
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizer


@dataclass
class PPOBatch:
    """Batch of PPO training data."""

    input_ids: torch.Tensor  # (batch_size, seq_len)
    attention_mask: torch.Tensor  # (batch_size, seq_len)
    actions: torch.Tensor  # (batch_size, seq_len) - generated tokens
    rewards: torch.Tensor  # (batch_size,) - episode rewards
    values: torch.Tensor  # (batch_size, seq_len) - value estimates
    advantages: torch.Tensor  # (batch_size, seq_len) - computed advantages
    old_log_probs: torch.Tensor  # (batch_size, seq_len) - old policy log probs


@dataclass
class DPOBatch:
    """Batch of DPO training data."""

    # Chosen completions
    chosen_input_ids: torch.Tensor  # (batch_size, seq_len)
    chosen_attention_mask: torch.Tensor  # (batch_size, seq_len)
    chosen_labels: torch.Tensor  # (batch_size, seq_len)

    # Rejected completions
    rejected_input_ids: torch.Tensor  # (batch_size, seq_len)
    rejected_attention_mask: torch.Tensor  # (batch_size, seq_len)
    rejected_labels: torch.Tensor  # (batch_size, seq_len)

    # Optional metadata
    confidence: Optional[torch.Tensor] = None  # (batch_size,)


class PPODataCollator:
    """Collator for PPO rollout batches."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: str = "max_length",
    ):
        """
        Initialize PPO data collator.

        Args:
            tokenizer: Tokenizer for padding
            max_length: Maximum sequence length
            padding: Padding strategy
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def __call__(self, examples: List[Dict[str, Any]]) -> PPOBatch:
        """
        Collate PPO examples into batch.

        Args:
            examples: List of dicts with keys:
                - query: str
                - response: str
                - reward: float
                - value: List[float]
                - advantage: List[float]
                - old_log_prob: List[float]

        Returns:
            PPOBatch
        """
        # Extract queries and responses
        queries = [ex['query'] for ex in examples]
        responses = [ex['response'] for ex in examples]

        # Tokenize queries
        query_encodings = self.tokenizer(
            queries,
            max_length=self.max_length // 2,
            padding=self.padding,
            truncation=True,
            return_tensors='pt',
        )

        # Tokenize responses (these are the "actions")
        response_encodings = self.tokenizer(
            responses,
            max_length=self.max_length // 2,
            padding=self.padding,
            truncation=True,
            return_tensors='pt',
        )

        # Concatenate query + response
        input_ids = torch.cat([
            query_encodings['input_ids'],
            response_encodings['input_ids']
        ], dim=1)

        attention_mask = torch.cat([
            query_encodings['attention_mask'],
            response_encodings['attention_mask']
        ], dim=1)

        # Pad to max_length if needed
        if input_ids.shape[1] < self.max_length:
            padding_length = self.max_length - input_ids.shape[1]
            input_ids = torch.nn.functional.pad(
                input_ids, (0, padding_length), value=self.tokenizer.pad_token_id
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, padding_length), value=0
            )
        elif input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]

        # Actions are the response tokens
        actions = response_encodings['input_ids']

        # Extract rewards, values, advantages, old_log_probs
        rewards = torch.tensor([ex['reward'] for ex in examples], dtype=torch.float32)

        # Pad values, advantages, old_log_probs to match sequence length
        max_seq_len = input_ids.shape[1]

        def pad_sequence_values(values_list, target_len):
            """Pad list of values to target length."""
            padded = []
            for vals in values_list:
                vals_tensor = torch.tensor(vals, dtype=torch.float32)
                if len(vals_tensor) < target_len:
                    vals_tensor = torch.nn.functional.pad(
                        vals_tensor, (0, target_len - len(vals_tensor)), value=0.0
                    )
                elif len(vals_tensor) > target_len:
                    vals_tensor = vals_tensor[:target_len]
                padded.append(vals_tensor)
            return torch.stack(padded)

        values = pad_sequence_values([ex['value'] for ex in examples], max_seq_len)
        advantages = pad_sequence_values([ex['advantage'] for ex in examples], max_seq_len)
        old_log_probs = pad_sequence_values([ex['old_log_prob'] for ex in examples], max_seq_len)

        return PPOBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            actions=actions,
            rewards=rewards,
            values=values,
            advantages=advantages,
            old_log_probs=old_log_probs,
        )


class DPODataCollator:
    """Collator for DPO preference pair batches."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ):
        """
        Initialize DPO data collator.

        Args:
            tokenizer: Tokenizer
            max_length: Maximum total sequence length
            max_prompt_length: Maximum prompt length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def __call__(self, examples: List[Dict[str, Any]]) -> DPOBatch:
        """
        Collate DPO examples into batch.

        Args:
            examples: List of dicts with keys:
                - prompt: str
                - chosen: str
                - rejected: str
                - confidence: float (optional)

        Returns:
            DPOBatch
        """
        prompts = [ex['prompt'] for ex in examples]
        chosen = [ex['chosen'] for ex in examples]
        rejected = [ex['rejected'] for ex in examples]

        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts,
            max_length=self.max_prompt_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Tokenize chosen completions
        chosen_encodings = self.tokenizer(
            chosen,
            max_length=self.max_length - self.max_prompt_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Tokenize rejected completions
        rejected_encodings = self.tokenizer(
            rejected,
            max_length=self.max_length - self.max_prompt_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Concatenate prompt + chosen
        chosen_input_ids = torch.cat([
            prompt_encodings['input_ids'],
            chosen_encodings['input_ids']
        ], dim=1)

        chosen_attention_mask = torch.cat([
            prompt_encodings['attention_mask'],
            chosen_encodings['attention_mask']
        ], dim=1)

        # Concatenate prompt + rejected
        rejected_input_ids = torch.cat([
            prompt_encodings['input_ids'],
            rejected_encodings['input_ids']
        ], dim=1)

        rejected_attention_mask = torch.cat([
            prompt_encodings['attention_mask'],
            rejected_encodings['attention_mask']
        ], dim=1)

        # Create labels (mask prompt, keep completion)
        prompt_len = prompt_encodings['input_ids'].shape[1]

        chosen_labels = chosen_input_ids.clone()
        chosen_labels[:, :prompt_len] = -100  # Ignore prompt in loss

        rejected_labels = rejected_input_ids.clone()
        rejected_labels[:, :prompt_len] = -100  # Ignore prompt in loss

        # Extract confidence if available
        confidence = None
        if 'confidence' in examples[0]:
            confidence = torch.tensor(
                [ex.get('confidence', 1.0) for ex in examples],
                dtype=torch.float32
            )

        return DPOBatch(
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=chosen_attention_mask,
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=rejected_attention_mask,
            rejected_labels=rejected_labels,
            confidence=confidence,
        )


def create_mock_ppo_batch(batch_size: int = 4, seq_len: int = 128) -> PPOBatch:
    """Create mock PPO batch for testing."""
    return PPOBatch(
        input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
        attention_mask=torch.ones(batch_size, seq_len),
        actions=torch.randint(0, 1000, (batch_size, seq_len // 2)),
        rewards=torch.randn(batch_size),
        values=torch.randn(batch_size, seq_len),
        advantages=torch.randn(batch_size, seq_len),
        old_log_probs=torch.randn(batch_size, seq_len),
    )


def create_mock_dpo_batch(batch_size: int = 4, seq_len: int = 128) -> DPOBatch:
    """Create mock DPO batch for testing."""
    return DPOBatch(
        chosen_input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
        chosen_attention_mask=torch.ones(batch_size, seq_len),
        chosen_labels=torch.randint(0, 1000, (batch_size, seq_len)),
        rejected_input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
        rejected_attention_mask=torch.ones(batch_size, seq_len),
        rejected_labels=torch.randint(0, 1000, (batch_size, seq_len)),
        confidence=torch.ones(batch_size),
    )
