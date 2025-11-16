"""
Value head for PPO training.

Provides learned value function for advantage estimation instead of mock values.
"""

import torch
import torch.nn as nn
from typing import Optional
from loguru import logger


class ValueHead(nn.Module):
    """
    Value function head for PPO.

    Predicts state values for advantage estimation.
    Can be attached to any transformer model.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        num_hidden_layers: int = 1,
    ):
        """
        Initialize value head.

        Args:
            hidden_size: Hidden size of base model
            dropout: Dropout probability
            num_hidden_layers: Number of hidden layers in value head
        """
        super().__init__()

        self.hidden_size = hidden_size

        # Build value network
        if num_hidden_layers == 1:
            # Simple linear projection
            self.value_net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
        else:
            # Multi-layer value network
            layers = []
            for i in range(num_hidden_layers):
                if i == 0:
                    layers.extend([
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ])
                elif i == num_hidden_layers - 1:
                    layers.append(nn.Linear(hidden_size // 2, 1))
                else:
                    layers.extend([
                        nn.Linear(hidden_size // 2, hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ])

            self.value_net = nn.Sequential(*layers)

        logger.info(f"Value head initialized: {hidden_size} -> 1, {num_hidden_layers} layers")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute values.

        Args:
            hidden_states: Hidden states from model (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Values (batch_size, seq_len)
        """
        # hidden_states: (batch_size, seq_len, hidden_size)
        values = self.value_net(hidden_states).squeeze(-1)  # (batch_size, seq_len)

        # Mask padding positions
        if attention_mask is not None:
            values = values * attention_mask

        return values


class ModelWithValueHead(nn.Module):
    """
    Wrapper that adds a value head to any model.

    Used for PPO training where we need both policy (model) and value function.
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        value_head_dropout: float = 0.1,
        value_head_layers: int = 1,
    ):
        """
        Initialize model with value head.

        Args:
            model: Base model (e.g., PEFT model with LoRA)
            hidden_size: Hidden size of base model
            value_head_dropout: Dropout for value head
            value_head_layers: Number of layers in value head
        """
        super().__init__()

        self.model = model
        self.value_head = ValueHead(
            hidden_size=hidden_size,
            dropout=value_head_dropout,
            num_hidden_layers=value_head_layers,
        )

        self.hidden_size = hidden_size

        logger.info("Model wrapped with value head")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        Forward pass through model and value head.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_dict: Whether to return dict
            **kwargs: Additional arguments for model

        Returns:
            Dict with 'logits', 'values', and optionally 'hidden_states'
        """
        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        # Get last hidden states
        # outputs.hidden_states is a tuple of (num_layers,)
        # We want the last layer: (batch_size, seq_len, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]

        # Compute values
        values = self.value_head(last_hidden_states, attention_mask)

        if return_dict:
            return {
                'logits': outputs.logits,
                'values': values,
                'hidden_states': outputs.hidden_states,
            }
        else:
            return outputs.logits, values

    def generate(self, *args, **kwargs):
        """Forward generate calls to base model."""
        return self.model.generate(*args, **kwargs)

    def save_pretrained(self, save_path, **kwargs):
        """Save model and value head."""
        # Save base model (includes LoRA)
        self.model.save_pretrained(save_path, **kwargs)

        # Save value head separately
        import os
        value_head_path = os.path.join(save_path, "value_head.pt")
        torch.save(self.value_head.state_dict(), value_head_path)

        logger.info(f"Saved model with value head to {save_path}")

    def load_value_head(self, load_path):
        """Load value head weights."""
        import os
        value_head_path = os.path.join(load_path, "value_head.pt")

        if os.path.exists(value_head_path):
            self.value_head.load_state_dict(torch.load(value_head_path))
            logger.success(f"Loaded value head from {value_head_path}")
        else:
            logger.warning(f"Value head weights not found at {value_head_path}")


def create_model_with_value_head(
    model: nn.Module,
    hidden_size: Optional[int] = None,
    value_head_dropout: float = 0.1,
    value_head_layers: int = 1,
) -> ModelWithValueHead:
    """
    Helper function to create model with value head.

    Args:
        model: Base model
        hidden_size: Hidden size (auto-detected if None)
        value_head_dropout: Dropout for value head
        value_head_layers: Number of layers in value head

    Returns:
        ModelWithValueHead instance
    """
    # Auto-detect hidden size if not provided
    if hidden_size is None:
        if hasattr(model, 'config'):
            hidden_size = model.config.hidden_size
        else:
            # Try to infer from model structure
            for name, param in model.named_parameters():
                if 'embed' in name.lower():
                    hidden_size = param.shape[-1]
                    break

            if hidden_size is None:
                raise ValueError("Could not auto-detect hidden_size, please provide it explicitly")

    logger.info(f"Creating model with value head (hidden_size={hidden_size})")

    return ModelWithValueHead(
        model=model,
        hidden_size=hidden_size,
        value_head_dropout=value_head_dropout,
        value_head_layers=value_head_layers,
    )


# Example usage:
if __name__ == "__main__":
    # Test value head
    batch_size = 4
    seq_len = 128
    hidden_size = 768

    # Mock hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)

    # Create value head
    value_head = ValueHead(hidden_size=hidden_size)

    # Forward pass
    values = value_head(hidden_states, attention_mask)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output values shape: {values.shape}")
    print(f"Values range: [{values.min():.3f}, {values.max():.3f}]")
