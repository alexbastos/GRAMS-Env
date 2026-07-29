"""Funções utilitárias compartilhadas entre agentes GNN e MLP.

Converte observações Gymnasium Dict → tensores PyTorch e
adjacency matrix densa → edge_index COO para PyTorch Geometric.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def obs_to_tensors(
    obs: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Converte observação Gymnasium Dict → tensores PyTorch.

    Parameters
    ----------
    obs : dict[str, np.ndarray]
        Observação com 'node_features' (V, 3) e 'adjacency_matrix' (V, V).
    device : torch.device
        Dispositivo de destino (cpu/cuda).

    Returns
    -------
    dict[str, torch.Tensor]
        Dicionário com tensores no dispositivo especificado.
    """
    return {
        key: torch.as_tensor(val, dtype=torch.float32, device=device)
        for key, val in obs.items()
    }


def adj_to_edge_index(
    adj_matrix: torch.Tensor,
    threshold: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converte adjacency matrix densa → edge_index COO + edge_attr.

    Cria arestas para todos os pares (i, j) onde adj[i,j] > threshold,
    incluindo self-loops (diagonal).

    Parameters
    ----------
    adj_matrix : torch.Tensor
        Matriz de adjacência densa (V, V).
    threshold : float
        Limiar mínimo para considerar uma aresta.

    Returns
    -------
    edge_index : torch.Tensor
        Índices COO (2, E) das arestas.
    edge_attr : torch.Tensor
        Atributos das arestas (E, 1) — ganho de interferência.
    """
    # Adiciona self-loops com valor 1.0 na diagonal
    v = adj_matrix.size(0)
    adj_with_self = adj_matrix.clone()
    adj_with_self[range(v), range(v)] = torch.clamp(
        adj_with_self.diagonal(), min=1.0,
    )

    # Encontra arestas acima do threshold
    row, col = torch.where(adj_with_self > threshold)
    edge_index = torch.stack([row, col], dim=0)  # (2, E)
    edge_attr = adj_with_self[row, col].unsqueeze(-1)  # (E, 1)

    return edge_index, edge_attr


def set_seed(seed: int) -> None:
    """Fixa seeds para reprodutibilidade.

    Parameters
    ----------
    seed : int
        Semente aleatória.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
