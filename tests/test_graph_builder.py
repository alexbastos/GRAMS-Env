"""Testes unitários para o adaptador de grafo — SEM gymnasium."""

import numpy as np
import pytest

from grams_env.adapters.graph_builder import GraphBuilder
from grams_env.core.domain.network_state import NetworkState


class TestGraphBuilder:
    """Testa a conversão de NetworkState → GraphObservation."""

    def _make_state(self, num_ues: int = 5) -> NetworkState:
        """Cria um NetworkState de teste."""
        rng = np.random.default_rng(42)
        return NetworkState(
            positions=rng.uniform(-500, 500, (num_ues, 2)),
            speeds=rng.uniform(0, 5, num_ues),
            directions=rng.uniform(0, 2 * np.pi, num_ues),
            queues=rng.uniform(0, 10000, num_ues),
            cbr_bytes=rng.choice([1000.0, 4000.0], num_ues),
            direct_gains=rng.exponential(1e-10, num_ues),
            interference_gains=rng.exponential(1e-15, (num_ues, num_ues)),
            is_los=rng.choice([True, False], num_ues),
        )

    def test_output_shapes(self):
        """Shapes devem ser (V, 3) e (V, V)."""
        builder = GraphBuilder()
        state = self._make_state(num_ues=10)
        obs = builder.build(state)
        assert obs.node_features.shape == (10, 3)
        assert obs.adjacency_matrix.shape == (10, 10)

    def test_output_dtype(self):
        """Tipos devem ser float32."""
        builder = GraphBuilder()
        state = self._make_state()
        obs = builder.build(state)
        assert obs.node_features.dtype == np.float32
        assert obs.adjacency_matrix.dtype == np.float32

    def test_node_features_content(self):
        """node_features deve conter CQI, queue, cbr."""
        builder = GraphBuilder()
        state = self._make_state(num_ues=3)
        obs = builder.build(state)

        # Coluna 1 é a fila
        np.testing.assert_allclose(
            obs.node_features[:, 1],
            state.queues.astype(np.float32),
            rtol=1e-5,
        )

        # Coluna 2 é o CBR
        np.testing.assert_allclose(
            obs.node_features[:, 2],
            state.cbr_bytes.astype(np.float32),
            rtol=1e-5,
        )

    def test_to_dict(self):
        """to_dict deve retornar dict com as chaves corretas."""
        builder = GraphBuilder()
        state = self._make_state()
        obs = builder.build(state)
        d = builder.to_dict(obs)
        assert "node_features" in d
        assert "adjacency_matrix" in d
        assert isinstance(d["node_features"], np.ndarray)
        assert isinstance(d["adjacency_matrix"], np.ndarray)

    def test_observation_is_frozen(self):
        """GraphObservation deve ser frozen (imutável)."""
        builder = GraphBuilder()
        state = self._make_state()
        obs = builder.build(state)
        with pytest.raises(AttributeError):
            obs.node_features = np.zeros((5, 3))
