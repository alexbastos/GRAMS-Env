"""Testes dos modelos Actor-Critic (GNN e MLP) — shapes, gradientes, generalização."""

import numpy as np
import pytest
import torch

from grams_env.agents.gnn.gnn_actor_critic import GNNActorCritic
from grams_env.agents.mlp.mlp_actor_critic import MLPActorCritic


# ====================================================================== #
#  Fixtures                                                                #
# ====================================================================== #

def _make_obs(num_ues: int) -> dict[str, np.ndarray]:
    """Cria observação sintética para testes."""
    return {
        "node_features": np.random.randn(num_ues, 3).astype(np.float32),
        "adjacency_matrix": np.abs(
            np.random.randn(num_ues, num_ues)
        ).astype(np.float32),
    }


# ====================================================================== #
#  GNNActorCritic                                                          #
# ====================================================================== #

class TestGNNActorCriticAct:
    """Testa GNNActorCritic.act() — inferência."""

    @pytest.fixture
    def model(self) -> GNNActorCritic:
        return GNNActorCritic(
            in_features=3, hidden_dim=32, num_layers=2,
            num_heads=2, num_rbs=50,
        )

    def test_act_shapes(self, model: GNNActorCritic):
        """act() deve retornar (action, log_prob, value) com shapes corretas."""
        obs = _make_obs(10)
        action, log_prob, value = model.act(obs)

        assert action.shape == (50,), f"action shape: {action.shape}"
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_act_action_range(self, model: GNNActorCritic):
        """Ações devem estar no range [0, V-1]."""
        obs = _make_obs(10)
        action, _, _ = model.act(obs)

        assert np.all(action >= 0)
        assert np.all(action < 10)

    def test_act_different_v(self, model: GNNActorCritic):
        """GNN deve funcionar com V diferente (zero-shot)."""
        for v in [5, 10, 20, 50]:
            obs = _make_obs(v)
            action, log_prob, value = model.act(obs)
            assert action.shape == (50,)
            assert np.all(action >= 0)
            assert np.all(action < v)


class TestGNNActorCriticEvaluate:
    """Testa GNNActorCritic.evaluate() — training."""

    @pytest.fixture
    def model(self) -> GNNActorCritic:
        return GNNActorCritic(
            in_features=3, hidden_dim=32, num_layers=2,
            num_heads=2, num_rbs=50,
        )

    def test_evaluate_shapes(self, model: GNNActorCritic):
        """evaluate() deve retornar tensores com shape (B,)."""
        batch_size = 4
        obs_nf = [torch.randn(10, 3) for _ in range(batch_size)]
        obs_adj = [torch.rand(10, 10) for _ in range(batch_size)]
        actions = torch.randint(0, 10, (batch_size, 50))

        log_probs, values, entropy = model.evaluate(
            obs_nf, obs_adj, actions,
        )

        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size,)
        assert entropy.shape == (batch_size,)

    def test_evaluate_gradient_flow(self, model: GNNActorCritic):
        """Gradientes devem fluir de evaluate() para todos os parâmetros."""
        obs_nf = [torch.randn(10, 3)]
        obs_adj = [torch.rand(10, 10)]
        actions = torch.randint(0, 10, (1, 50))

        log_probs, values, entropy = model.evaluate(
            obs_nf, obs_adj, actions,
        )
        loss = -log_probs.sum() + values.sum()
        loss.backward()

        trainable_params = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad
        ]
        assert len(trainable_params) > 0
        for name, param in trainable_params:
            assert param.grad is not None, f"Grad None: {name}"


# ====================================================================== #
#  MLPActorCritic                                                          #
# ====================================================================== #

class TestMLPActorCriticAct:
    """Testa MLPActorCritic.act() — inferência."""

    @pytest.fixture
    def model(self) -> MLPActorCritic:
        return MLPActorCritic(
            num_ues=10, num_rbs=50, hidden_dim=64,
        )

    def test_act_shapes(self, model: MLPActorCritic):
        """act() deve retornar shapes corretas."""
        obs = _make_obs(10)
        action, log_prob, value = model.act(obs)

        assert action.shape == (50,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_act_action_range(self, model: MLPActorCritic):
        """Ações devem estar no range [0, V-1]."""
        obs = _make_obs(10)
        action, _, _ = model.act(obs)

        assert np.all(action >= 0)
        assert np.all(action < 10)


class TestMLPActorCriticEvaluate:
    """Testa MLPActorCritic.evaluate() — training."""

    @pytest.fixture
    def model(self) -> MLPActorCritic:
        return MLPActorCritic(
            num_ues=10, num_rbs=50, hidden_dim=64,
        )

    def test_evaluate_shapes(self, model: MLPActorCritic):
        """evaluate() deve retornar tensores com shape (B,)."""
        batch_size = 4
        obs_nf = [torch.randn(10, 3) for _ in range(batch_size)]
        obs_adj = [torch.rand(10, 10) for _ in range(batch_size)]
        actions = torch.randint(0, 10, (batch_size, 50))

        log_probs, values, entropy = model.evaluate(
            obs_nf, obs_adj, actions,
        )

        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size,)
        assert entropy.shape == (batch_size,)

    def test_evaluate_gradient_flow(self, model: MLPActorCritic):
        """Gradientes devem fluir pelo MLP."""
        obs_nf = [torch.randn(10, 3)]
        obs_adj = [torch.rand(10, 10)]
        actions = torch.randint(0, 10, (1, 50))

        log_probs, values, entropy = model.evaluate(
            obs_nf, obs_adj, actions,
        )
        loss = -log_probs.sum() + values.sum()
        loss.backward()

        trainable_params = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad
        ]
        assert len(trainable_params) > 0


class TestMLPGeneralizationFailure:
    """Prova que o MLP NÃO generaliza para V diferente."""

    def test_mlp_fails_with_different_v(self):
        """MLP treinado com V=10 deve falhar com V=20."""
        model = MLPActorCritic(num_ues=10, num_rbs=50, hidden_dim=64)
        obs_wrong_v = _make_obs(20)  # V=20, mas modelo espera V=10

        with pytest.raises((RuntimeError, ValueError)):
            model.act(obs_wrong_v)

    def test_gnn_succeeds_with_different_v(self):
        """Contraste: GNN deve funcionar com qualquer V."""
        model = GNNActorCritic(
            in_features=3, hidden_dim=32, num_layers=2,
            num_heads=2, num_rbs=50,
        )
        for v in [5, 10, 20, 50]:
            obs = _make_obs(v)
            action, _, _ = model.act(obs)
            assert action.shape == (50,)
