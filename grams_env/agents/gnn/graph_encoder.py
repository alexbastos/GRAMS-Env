"""Extrator de Características GNN (GATConv) — PyTorch Geometric.

Rede neural em grafos que transforma observações estruturadas
do GRAMS-Env (node_features + adjacency_matrix) em embeddings
por nó H^(L), invariantes ao número de UEs (V).

Arquitetura:
    Input (V, 3) → Linear(3, hidden) → [GATConv + ELU + LayerNorm] × L → H^(L) (V, hidden)

A invariância ao número de nós é garantida pelo message passing:
cada nó agrega informação dos vizinhos independentemente, permitindo
treinar com V=20 e inferir com V=50, 100, 200 (zero-shot).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GraphEncoder(nn.Module):
    """Extrator de features via Graph Attention Network (GAT).

    Parameters
    ----------
    in_features : int
        Dimensão das features de entrada por nó (default: 3 = CQI, queue, cbr).
    hidden_dim : int
        Dimensão dos embeddings intermediários e de saída.
    num_layers : int
        Número de camadas GATConv (L).
    num_heads : int
        Número de heads de atenção em cada camada GAT.
    edge_dim : int
        Dimensão dos atributos de aresta (1 = ganho de interferência).
    dropout : float
        Taxa de dropout para regularização.
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        edge_dim: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Projeção linear da entrada para hidden_dim
        self.input_proj = nn.Linear(in_features, hidden_dim)

        # Camadas GATConv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=False,       # Média dos heads → output = hidden_dim
                    edge_dim=edge_dim,  # Usa ganho de interferência como peso
                    dropout=dropout,
                    add_self_loops=False,  # Self-loops adicionados na conversão
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.activation = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass — gera embeddings por nó.

        Parameters
        ----------
        x : torch.Tensor
            Node features (V, in_features).
        edge_index : torch.Tensor
            Índices COO das arestas (2, E).
        edge_attr : torch.Tensor | None
            Atributos das arestas (E, edge_dim).

        Returns
        -------
        torch.Tensor
            Embeddings H^(L) por nó (V, hidden_dim).
        """
        # Projeção inicial
        h = self.input_proj(x)  # (V, hidden_dim)

        # Camadas de convolução em grafos
        for conv, norm in zip(self.convs, self.norms):
            h_res = h                                     # Residual connection
            h = conv(h, edge_index, edge_attr=edge_attr)  # GATConv
            h = self.activation(h)                         # ELU
            h = norm(h)                                    # LayerNorm
            h = h + h_res                                  # Residual

        return h  # (V, hidden_dim)
