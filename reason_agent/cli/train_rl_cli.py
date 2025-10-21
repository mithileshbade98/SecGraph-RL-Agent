#!/usr/bin/env python
"""CLI for RL training (PPO/DPO)."""

import click
from pathlib import Path
from loguru import logger


@click.command()
@click.option('--mode', type=click.Choice(['ppo', 'dpo']), required=True, help='Training mode')
@click.option('--config', help='Config file')
@click.option('--episodes', type=int, default=100, help='Episodes (PPO)')
@click.option('--epochs', type=int, default=1, help='Epochs (DPO)')
@click.option('--data', help='Data path (DPO)')
def main(mode, config, episodes, epochs, data):
    """Train RL models (PPO or DPO)."""

    if mode == 'ppo':
        from reason_agent.rl.ppo import PPOTrainer

        config_path = Path(config) if config else Path('configs/rl/ppo.yaml')
        trainer = PPOTrainer(config_path=config_path)

        logger.info(f"Training PPO for {episodes} episodes...")
        metrics = trainer.train(num_episodes=episodes)

        logger.info(f"✅ PPO training complete!")
        logger.info(f"Final reward: {metrics['final_avg_reward']:.3f}")

    elif mode == 'dpo':
        from reason_agent.rl.dpo import DPOTrainer

        config_path = Path(config) if config else Path('configs/rl/dpo.yaml')
        trainer = DPOTrainer(config_path=config_path)

        data_path = Path(data) if data else None

        logger.info(f"Training DPO for {epochs} epochs...")
        metrics = trainer.train(data_path=data_path, num_epochs=epochs)

        logger.info(f"✅ DPO training complete!")
        logger.info(f"Final accuracy: {metrics['final_accuracy']:.3f}")
        logger.info(f"Trained on {metrics['num_pairs']} pairs")


if __name__ == '__main__':
    main()
