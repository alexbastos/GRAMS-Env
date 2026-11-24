"""Normalização de features da GNN e compatibilidade de checkpoints legados."""

import numpy as np
import pytest
import torch

from grams_env.agents.gnn.gnn_actor_critic import GNNActorCritic


def _make_model(**kwargs):
    return GNNActorCritic(
        hidden_dim=16, num_layers=1, num_heads=2, num_rbs=4, **kwargs,
    ).eval()


@pytest.fixture
def model():
    return _make_model()


@pytest.fixture
def obs():
    return {
        "node_features": np.array([
            [-20, -100, -40],
            [0, 0, 0],
            [100, 320000, 2000],
            [200, 960000, 4000],
            [400, 3200000, 8000],
        ], dtype=np.float32),
        "adjacency_matrix": np.ones((5, 5), dtype=np.float32),
    }


def test_default_buffers(model):
    buffers = dict(model.named_buffers())
    assert buffers["feature_normalization_enabled"].dtype == torch.bool
    assert buffers["feature_normalization_enabled"].item() is True
    torch.testing.assert_close(
        buffers["feature_scales"], torch.tensor([200., 320000., 4000.]),
    )


def test_normalization_values_and_limits(model, obs):
    features = torch.from_numpy(obs["node_features"])
    original = features.clone()
    normalized = model.normalize_node_features(features)
    expected = torch.tensor([
        [0, 0, 0],
        [0, 0, 0],
        [0.5, 0.5, 0.5],
        [1, 0.75, 1],
        [1, 10 / 11, 1],
    ])
    torch.testing.assert_close(normalized, expected)
    assert torch.isfinite(normalized).all()
    assert ((normalized >= 0) & (normalized <= 1)).all()
    torch.testing.assert_close(features, original, rtol=0, atol=0)


def test_queue_is_strictly_increasing_without_capacity_clipping(model):
    queues = torch.tensor([0., 1., 32000., 320000., 640000., 3200000., 3.2e9])
    features = torch.zeros(len(queues), 3)
    features[:, 1] = queues
    normalized = model.normalize_node_features(features)[:, 1]
    torch.testing.assert_close(normalized, queues / (queues + 320000))
    assert torch.all(torch.diff(normalized) > 0)
    assert torch.all(normalized < 1)


def test_disabled_normalization_preserves_raw_features(obs):
    model = _make_model(normalize_features=False)
    features = torch.from_numpy(obs["node_features"])
    original = features.clone()
    torch.testing.assert_close(
        model.normalize_node_features(features), original, rtol=0, atol=0,
    )
    torch.testing.assert_close(features, original, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["act", "evaluate", "get_value"])
def test_public_paths_normalize_encoder_input_without_mutating_obs(model, obs, method):
    original = {key: value.copy() for key, value in obs.items()}
    expected = model.normalize_node_features(torch.from_numpy(obs["node_features"]))
    captured = []

    def capture_input(module, args):
        captured.append(args[0].detach().clone())

    handle = model.encoder.register_forward_pre_hook(capture_input)
    try:
        if method == "evaluate":
            model.evaluate(
                [torch.from_numpy(obs["node_features"])],
                [torch.from_numpy(obs["adjacency_matrix"])],
                torch.tensor([[0, 1, 2, 3]]),
            )
        else:
            getattr(model, method)(obs)
    finally:
        handle.remove()

    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected)
    for key in obs:
        np.testing.assert_array_equal(obs[key], original[key])


def test_normalization_has_finite_nonzero_interior_gradients(model, obs):
    features = torch.tensor(obs["node_features"], requires_grad=True)
    model.normalize_node_features(features).sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    torch.testing.assert_close(
        features.grad[2], torch.tensor([1 / 200, 1 / 1280000, 1 / 4000]),
    )
    torch.testing.assert_close(features.grad[0], torch.zeros(3))
    assert features.grad[-1, 0] == 0
    assert features.grad[-1, 2] == 0


def test_evaluate_has_finite_gradients(model, obs):
    features = torch.tensor(obs["node_features"], requires_grad=True)
    outputs = model.evaluate(
        [features], [torch.from_numpy(obs["adjacency_matrix"])],
        torch.tensor([[0, 1, 2, 3]]),
    )
    log_probs, values, entropy = outputs
    assert all(torch.isfinite(output).all() for output in outputs)
    (-log_probs.mean() + values.square().mean() - 0.01 * entropy.mean()).backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("enabled", [True, False])
def test_state_dict_roundtrip_preserves_scales_and_flag(tmp_path, obs, enabled):
    source = _make_model(normalize_features=enabled)
    source.feature_scales.copy_(torch.tensor([100., 640000., 8000.]))
    checkpoint = tmp_path / "normalized.pt"
    torch.save(source.state_dict(), checkpoint)
    state = torch.load(checkpoint, weights_only=True)
    assert state["feature_normalization_enabled"].dtype == torch.bool
    assert state["feature_normalization_enabled"].item() is enabled
    torch.testing.assert_close(state["feature_scales"], source.feature_scales)

    restored = _make_model(normalize_features=not enabled)
    restored.load_state_dict(state, strict=True)
    assert restored.feature_normalization_enabled.item() is enabled
    torch.testing.assert_close(restored.feature_scales, source.feature_scales)
    features = torch.from_numpy(obs["node_features"])
    adjacency = torch.from_numpy(obs["adjacency_matrix"])
    torch.testing.assert_close(
        restored.normalize_node_features(features), source.normalize_node_features(features),
    )
    with torch.no_grad():
        expected_dist, expected_value = source._get_policy_and_value(features, adjacency)
        actual_dist, actual_value = restored._get_policy_and_value(features, adjacency)
    torch.testing.assert_close(actual_dist.logits, expected_dist.logits)
    torch.testing.assert_close(actual_value, expected_value)


def test_legacy_checkpoint_preserves_unnormalized_logits_and_value(obs):
    source = _make_model(normalize_features=False)
    legacy_state = source.state_dict()
    del legacy_state["feature_normalization_enabled"]
    del legacy_state["feature_scales"]
    restored = _make_model()
    assert restored.feature_normalization_enabled.item() is True
    result = restored.load_state_dict(legacy_state, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert restored.feature_normalization_enabled.item() is False
    features = torch.from_numpy(obs["node_features"])
    adjacency = torch.from_numpy(obs["adjacency_matrix"])
    torch.testing.assert_close(restored.normalize_node_features(features), features)
    with torch.no_grad():
        expected_dist, expected_value = source._get_policy_and_value(features, adjacency)
        actual_dist, actual_value = restored._get_policy_and_value(features, adjacency)
    torch.testing.assert_close(actual_dist.logits, expected_dist.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    assert restored.get_value(obs) == source.get_value(obs)
