"""Smoke test do PPO — treina 2 iterações, verifica que não há crash."""

import pytest
import torch

from grams_env.agents.common.ppo_trainer import PPOConfig, PPOTrainer
from grams_env.agents.gnn.gnn_actor_critic import GNNActorCritic
from grams_env.agents.mlp.mlp_actor_critic import MLPActorCritic
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env


class TestPPOSmokeGNN:
    """Smoke test: PPO + GNN sem crash."""

    def test_train_2_iterations(self, tmp_path):
        """Deve treinar 2 iterações sem erros."""
        env = OpenRAN_RBA_Env(num_ues=5, max_steps=32)

        policy = GNNActorCritic(
            in_features=3, hidden_dim=16, num_layers=1,
            num_heads=2, num_rbs=env.num_rbs,
        )

        config = PPOConfig(
            lr=1e-3,
            rollout_steps=32,
            epochs=2,
            batch_size=16,
            total_iterations=2,
            checkpoint_interval=10,
            device="cpu",
        )

        trainer = PPOTrainer(
            policy=policy, env=env, config=config,
            save_dir=tmp_path / "gnn_smoke",
        )

        # Não deve levantar exceções
        trainer.train()

        # Checkpoint deve existir
        assert (tmp_path / "gnn_smoke" / "checkpoint_final.pt").exists()
        assert (tmp_path / "gnn_smoke" / "training_log.csv").exists()


class TestPPOSmokeMLP:
    """Smoke test: PPO + MLP sem crash."""

    def test_train_2_iterations(self, tmp_path):
        """Deve treinar 2 iterações sem erros."""
        env = OpenRAN_RBA_Env(num_ues=5, max_steps=32)

        policy = MLPActorCritic(
            num_ues=5, num_rbs=env.num_rbs, hidden_dim=32,
        )

        config = PPOConfig(
            lr=1e-3,
            rollout_steps=32,
            epochs=2,
            batch_size=16,
            total_iterations=2,
            checkpoint_interval=10,
            device="cpu",
        )

        trainer = PPOTrainer(
            policy=policy, env=env, config=config,
            save_dir=tmp_path / "mlp_smoke",
        )

        # Não deve levantar exceções
        trainer.train()

        # Checkpoint deve existir
        assert (tmp_path / "mlp_smoke" / "checkpoint_final.pt").exists()
        assert (tmp_path / "mlp_smoke" / "training_log.csv").exists()
