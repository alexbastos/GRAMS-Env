"""Testes do RolloutBuffer — inserção, GAE, mini-batches."""

import numpy as np
import pytest
import torch

from grams_env.agents.common.rollout_buffer import RolloutBuffer


class TestRolloutBufferAdd:
    """Testa inserção de transições."""

    def test_add_increments_size(self):
        """size deve incrementar a cada add()."""
        buf = RolloutBuffer(rollout_steps=64, num_rbs=50)
        assert buf.size == 0

        obs = {
            "node_features": np.zeros((10, 3), dtype=np.float32),
            "adjacency_matrix": np.zeros((10, 10), dtype=np.float32),
        }
        buf.add(obs, np.zeros(50, dtype=np.int64), 1.0, False, -0.5, 0.1)
        assert buf.size == 1

    def test_reset_clears_buffer(self):
        """reset() deve zerar o buffer."""
        buf = RolloutBuffer(rollout_steps=64, num_rbs=50)
        obs = {
            "node_features": np.zeros((10, 3), dtype=np.float32),
            "adjacency_matrix": np.zeros((10, 10), dtype=np.float32),
        }
        for _ in range(10):
            buf.add(obs, np.zeros(50, dtype=np.int64), 1.0, False, -0.5, 0.1)
        assert buf.size == 10

        buf.reset()
        assert buf.size == 0


class TestRolloutBufferGAE:
    """Testa cálculo de GAE."""

    def test_gae_shapes(self):
        """Advantages e returns devem ter tamanho n."""
        buf = RolloutBuffer(
            rollout_steps=16, num_rbs=50, gamma=0.99, gae_lambda=0.95,
        )
        obs = {
            "node_features": np.zeros((5, 3), dtype=np.float32),
            "adjacency_matrix": np.zeros((5, 5), dtype=np.float32),
        }

        for i in range(16):
            buf.add(
                obs, np.zeros(50, dtype=np.int64),
                reward=float(i), done=(i == 15),
                log_prob=-1.0, value=float(i) * 0.5,
            )

        buf.compute_gae(last_value=0.0)

        assert buf._advantages is not None
        assert buf._returns is not None
        assert len(buf._advantages) == 16
        assert len(buf._returns) == 16

    def test_gae_terminal_bootstrap(self):
        """Quando done=True, next_value deve ser 0 (sem bootstrap)."""
        buf = RolloutBuffer(
            rollout_steps=2, num_rbs=50, gamma=0.99, gae_lambda=0.95,
        )
        obs = {
            "node_features": np.zeros((5, 3), dtype=np.float32),
            "adjacency_matrix": np.zeros((5, 5), dtype=np.float32),
        }

        # Step 0: reward=1, done=False, value=0
        buf.add(obs, np.zeros(50, dtype=np.int64), 1.0, False, 0.0, 0.0)
        # Step 1: reward=2, done=True, value=0
        buf.add(obs, np.zeros(50, dtype=np.int64), 2.0, True, 0.0, 0.0)

        buf.compute_gae(last_value=100.0)  # last_value ignorado por done=True

        # Para step 1 (terminal): advantage = reward - value = 2 - 0 = 2
        assert abs(buf._advantages[1] - 2.0) < 1e-6

    def test_gae_no_nan(self):
        """GAE não deve gerar NaN."""
        buf = RolloutBuffer(rollout_steps=32, num_rbs=50)
        obs = {
            "node_features": np.random.randn(10, 3).astype(np.float32),
            "adjacency_matrix": np.random.rand(10, 10).astype(np.float32),
        }
        for _ in range(32):
            buf.add(
                obs, np.zeros(50, dtype=np.int64),
                reward=np.random.randn(), done=False,
                log_prob=-1.0, value=np.random.randn(),
            )
        buf.compute_gae(last_value=0.0)

        assert not np.any(np.isnan(buf._advantages))
        assert not np.any(np.isnan(buf._returns))


class TestRolloutBufferBatches:
    """Testa geração de mini-batches."""

    def test_batches_cover_all_samples(self):
        """Mini-batches devem cobrir todas as amostras."""
        n = 32
        batch_size = 8
        buf = RolloutBuffer(rollout_steps=n, num_rbs=50)
        obs = {
            "node_features": np.zeros((5, 3), dtype=np.float32),
            "adjacency_matrix": np.zeros((5, 5), dtype=np.float32),
        }
        for _ in range(n):
            buf.add(obs, np.zeros(50, dtype=np.int64), 1.0, False, -1.0, 0.5)
        buf.compute_gae(last_value=0.0)

        batches = buf.get_batches(batch_size)
        total_samples = sum(b["actions"].shape[0] for b in batches)
        assert total_samples == n

    def test_batch_keys(self):
        """Cada batch deve conter as chaves esperadas."""
        buf = RolloutBuffer(rollout_steps=8, num_rbs=50)
        obs = {
            "node_features": np.zeros((5, 3), dtype=np.float32),
            "adjacency_matrix": np.zeros((5, 5), dtype=np.float32),
        }
        for _ in range(8):
            buf.add(obs, np.zeros(50, dtype=np.int64), 1.0, False, -1.0, 0.5)
        buf.compute_gae(last_value=0.0)

        batches = buf.get_batches(4)
        expected_keys = {
            "obs_node_features", "obs_adj_matrix", "actions",
            "old_log_probs", "advantages", "returns",
        }
        for batch in batches:
            assert expected_keys == set(batch.keys())

    def test_get_batches_before_gae_raises(self):
        """get_batches() antes de compute_gae() deve falhar."""
        buf = RolloutBuffer(rollout_steps=8, num_rbs=50)
        obs = {
            "node_features": np.zeros((5, 3), dtype=np.float32),
            "adjacency_matrix": np.zeros((5, 5), dtype=np.float32),
        }
        for _ in range(8):
            buf.add(obs, np.zeros(50, dtype=np.int64), 1.0, False, -1.0, 0.5)

        with pytest.raises(AssertionError, match="compute_gae"):
            buf.get_batches(4)
