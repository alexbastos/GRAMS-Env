"""Testes do GraphEncoder (GATConv) — shapes, invariância, gradientes."""

import pytest
import torch

from grams_env.agents.common.utils import adj_to_edge_index
from grams_env.agents.gnn.graph_encoder import GraphEncoder


class TestGraphEncoderShapes:
    """Testa shapes de saída do GraphEncoder para diferentes V."""

    @pytest.fixture
    def encoder(self) -> GraphEncoder:
        return GraphEncoder(
            in_features=3,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )

    @pytest.mark.parametrize("num_ues", [5, 10, 20, 50, 100])
    def test_output_shape(self, encoder: GraphEncoder, num_ues: int):
        """Output deve ser (V, hidden_dim) para qualquer V."""
        x = torch.randn(num_ues, 3)
        adj = torch.rand(num_ues, num_ues)
        edge_index, edge_attr = adj_to_edge_index(adj)

        out = encoder(x, edge_index, edge_attr)

        assert out.shape == (num_ues, 64)
        assert out.dtype == torch.float32

    def test_output_no_nan(self, encoder: GraphEncoder):
        """Output não deve conter NaN."""
        x = torch.randn(10, 3)
        adj = torch.rand(10, 10)
        edge_index, edge_attr = adj_to_edge_index(adj)

        out = encoder(x, edge_index, edge_attr)

        assert not torch.isnan(out).any()


class TestGraphEncoderInvariance:
    """Testa invariância ao número de nós (zero-shot)."""

    def test_same_weights_different_v(self):
        """Mesmos pesos devem funcionar com V diferente do 'treino'."""
        encoder = GraphEncoder(in_features=3, hidden_dim=32, num_layers=2)

        # "Treina" com V=10 (forward pass)
        x10 = torch.randn(10, 3)
        adj10 = torch.rand(10, 10)
        ei10, ea10 = adj_to_edge_index(adj10)
        out10 = encoder(x10, ei10, ea10)
        assert out10.shape == (10, 32)

        # "Infere" com V=50 (mesmos pesos, sem re-treino)
        x50 = torch.randn(50, 3)
        adj50 = torch.rand(50, 50)
        ei50, ea50 = adj_to_edge_index(adj50)
        out50 = encoder(x50, ei50, ea50)
        assert out50.shape == (50, 32)

    def test_single_node(self):
        """Deve funcionar com V=1 (edge case)."""
        encoder = GraphEncoder(in_features=3, hidden_dim=16, num_layers=2)
        x = torch.randn(1, 3)
        adj = torch.ones(1, 1)
        ei, ea = adj_to_edge_index(adj)

        out = encoder(x, ei, ea)
        assert out.shape == (1, 16)


class TestGraphEncoderGradients:
    """Testa que os gradientes fluem corretamente."""

    def test_backward_pass(self):
        """Loss.backward() deve gerar gradientes não-zero."""
        encoder = GraphEncoder(in_features=3, hidden_dim=32, num_layers=2)
        x = torch.randn(10, 3)
        adj = torch.rand(10, 10)
        ei, ea = adj_to_edge_index(adj)

        out = encoder(x, ei, ea)
        loss = out.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"Gradiente None para {name}"
            assert param.grad.abs().sum() > 0, (
                f"Gradiente zero para {name}"
            )
