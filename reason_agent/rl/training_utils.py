"""
Training utilities for RL algorithms.

Provides:
- Generalized Advantage Estimation (GAE)
- PPO loss computation
- DPO loss computation
- Reward computation utilities
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from loguru import logger


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards at each step (batch_size, seq_len)
        values: Value estimates at each step (batch_size, seq_len)
        next_values: Value estimates for next states (batch_size, seq_len)
        dones: Done flags (batch_size, seq_len)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns)
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    # Compute GAE backwards through time
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = next_values[:, t]
        else:
            next_value = values[:, t + 1]

        # TD error: delta = r + gamma * V(s') - V(s)
        delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]

        # GAE: A = delta + gamma * lambda * A(t+1)
        last_gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * last_gae
        advantages[:, t] = last_gae

    # Returns: R = A + V
    returns = advantages + values

    return advantages, returns


def compute_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    Compute PPO clipped objective loss.

    Args:
        log_probs: New policy log probabilities (batch_size, seq_len)
        old_log_probs: Old policy log probabilities (batch_size, seq_len)
        advantages: Computed advantages (batch_size, seq_len)
        clip_range: PPO clipping range

    Returns:
        PPO loss scalar
    """
    # Compute probability ratios: r = π_new / π_old
    ratio = torch.exp(log_probs - old_log_probs)

    # Compute clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages

    # PPO loss is negative of the minimum (we want to maximize)
    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_range: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute value function loss.

    Args:
        values: Predicted values (batch_size, seq_len)
        returns: Target returns (batch_size, seq_len)
        clip_range: Optional clipping range for value loss

    Returns:
        Value loss scalar
    """
    if clip_range is not None:
        # Clipped value loss (same as PPO paper)
        value_loss_unclipped = (values - returns) ** 2
        value_clipped = values + torch.clamp(
            values - returns, -clip_range, clip_range
        )
        value_loss_clipped = (value_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
    else:
        # Standard MSE loss
        value_loss = F.mse_loss(values, returns)

    return value_loss


def compute_dpo_loss(
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    reference_chosen_log_probs: torch.Tensor,
    reference_rejected_log_probs: torch.Tensor,
    beta: float = 0.1,
    use_offset: bool = False,
    offset_margin: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute DPO (Direct Preference Optimization) loss.

    DPO Loss: -log(σ(β * [log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)]))

    Where:
    - y_w: chosen (winning) completion
    - y_l: rejected (losing) completion
    - π: policy model
    - π_ref: reference model
    - β: temperature parameter
    - σ: sigmoid function

    Args:
        policy_chosen_log_probs: Log probs from policy for chosen (batch_size,)
        policy_rejected_log_probs: Log probs from policy for rejected (batch_size,)
        reference_chosen_log_probs: Log probs from reference for chosen (batch_size,)
        reference_rejected_log_probs: Log probs from reference for rejected (batch_size,)
        beta: DPO temperature parameter (controls strength of KL penalty)
        use_offset: Whether to use offset-DPO (adds margin)
        offset_margin: Margin for offset-DPO

    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Compute log ratios for chosen and rejected
    chosen_log_ratio = policy_chosen_log_probs - reference_chosen_log_probs
    rejected_log_ratio = policy_rejected_log_probs - reference_rejected_log_probs

    # Compute preference logits
    logits = beta * (chosen_log_ratio - rejected_log_ratio)

    # Add offset margin if using offset-DPO
    if use_offset:
        logits = logits - offset_margin

    # DPO loss: -log(sigmoid(logits))
    # Equivalent to: log(1 + exp(-logits))
    loss = -F.logsigmoid(logits).mean()

    # Compute metrics
    with torch.no_grad():
        # Accuracy: how often policy prefers chosen over rejected
        accuracy = (logits > 0).float().mean()

        # Preference margin: average gap between chosen and rejected
        margin = (chosen_log_ratio - rejected_log_ratio).mean()

        # Reward estimates (implicit from log ratios)
        chosen_reward = beta * chosen_log_ratio.mean()
        rejected_reward = beta * rejected_log_ratio.mean()

    metrics = {
        'accuracy': accuracy,
        'margin': margin,
        'chosen_reward': chosen_reward,
        'rejected_reward': rejected_reward,
    }

    return loss, metrics


def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute log probabilities from logits and labels.

    Args:
        logits: Model logits (batch_size, seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len)
        attention_mask: Attention mask (batch_size, seq_len)

    Returns:
        Log probabilities summed over sequence (batch_size,)
    """
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Compute log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for target tokens
    # Shape: (batch_size, seq_len - 1)
    target_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask padding tokens
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous()
        target_log_probs = target_log_probs * shift_mask

        # Mask labels that are -100 (ignored in loss)
        valid_mask = (shift_labels != -100).float()
        target_log_probs = target_log_probs * valid_mask

        # Sum over sequence
        sequence_log_probs = target_log_probs.sum(dim=-1)
    else:
        sequence_log_probs = target_log_probs.sum(dim=-1)

    return sequence_log_probs


def compute_kl_divergence(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference.

    KL(π || π_ref) = E[log π - log π_ref]

    Args:
        policy_log_probs: Log probs from policy (batch_size,)
        reference_log_probs: Log probs from reference (batch_size,)

    Returns:
        Mean KL divergence
    """
    kl = (policy_log_probs - reference_log_probs).mean()
    return kl


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of policy distribution.

    H(π) = -E[log π]

    Args:
        logits: Model logits (batch_size, seq_len, vocab_size)

    Returns:
        Mean entropy
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy


class RewardComputer:
    """Compute rewards for reasoning tasks."""

    def __init__(
        self,
        correctness_weight: float = 1.0,
        length_penalty: float = 0.01,
        step_penalty: float = 0.05,
    ):
        """
        Initialize reward computer.

        Args:
            correctness_weight: Weight for correctness reward
            length_penalty: Penalty per token (encourages brevity)
            step_penalty: Penalty per reasoning step
        """
        self.correctness_weight = correctness_weight
        self.length_penalty = length_penalty
        self.step_penalty = step_penalty

    def compute_reward(
        self,
        output: str,
        expected_output: Optional[str] = None,
        is_correct: Optional[bool] = None,
        num_tokens: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> float:
        """
        Compute reward for a reasoning output.

        Args:
            output: Generated output text
            expected_output: Expected output (if available)
            is_correct: Whether output is correct (overrides comparison)
            num_tokens: Number of tokens in output
            num_steps: Number of reasoning steps

        Returns:
            Total reward
        """
        reward = 0.0

        # Correctness reward
        if is_correct is not None:
            correctness = 1.0 if is_correct else 0.0
        elif expected_output is not None:
            # Simple string match (can be made more sophisticated)
            correctness = 1.0 if output.strip() == expected_output.strip() else 0.0
        else:
            # No ground truth available
            correctness = 0.5

        reward += self.correctness_weight * correctness

        # Length penalty (encourage conciseness)
        if num_tokens is not None:
            reward -= self.length_penalty * num_tokens
        else:
            reward -= self.length_penalty * len(output.split())

        # Step penalty (encourage efficiency)
        if num_steps is not None:
            reward -= self.step_penalty * num_steps

        return reward

    def compute_batch_rewards(
        self,
        outputs: list[str],
        expected_outputs: Optional[list[str]] = None,
        is_correct: Optional[list[bool]] = None,
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of outputs.

        Args:
            outputs: List of generated outputs
            expected_outputs: List of expected outputs
            is_correct: List of correctness flags

        Returns:
            Tensor of rewards (batch_size,)
        """
        rewards = []
        for i, output in enumerate(outputs):
            expected = expected_outputs[i] if expected_outputs else None
            correct = is_correct[i] if is_correct else None

            reward = self.compute_reward(
                output=output,
                expected_output=expected,
                is_correct=correct,
            )
            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)


def clip_gradients(model, max_grad_norm: float = 1.0) -> float:
    """
    Clip gradients to prevent exploding gradients.

    Args:
        model: PyTorch model
        max_grad_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm
    )
    return total_norm.item()


def whiten_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize advantages to have mean 0 and std 1.

    Args:
        advantages: Advantage estimates
        eps: Small constant for numerical stability

    Returns:
        Whitened advantages
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)
