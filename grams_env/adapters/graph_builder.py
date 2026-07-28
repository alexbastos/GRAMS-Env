"""Adaptador: converte NetworkState → observação de grafo para GNN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grams_env.core.domain.network_state import NetworkState


@dataclass(frozen=True)
class GraphObservation:
    """Estrutura de grafo para consumo pela GNN.

    Desacoplada do gymnasium — pode ser consumida por
    PyTorch Geometric, DGL, ou qualquer outro framework.
    """

    node_features: np.ndarray       # (V, 3) float32
    adjacency_matrix: np.ndarray    # (V, V) float32


class GraphBuilder:
    """Converte um NetworkState em GraphObservation.

    Responsabilidade única: serialização de domínio → grafo.
    """

    def build(self, state: NetworkState) -> GraphObservation:
        """Constrói a observação de grafo a partir do estado da rede.

        node_features (V, 3):
            col 0 — CQI: proxy do ganho direto em escala dB normalizada.
            col 1 — Tamanho da fila atual (bits).
            col 2 — Carga de tráfego alvo CBR (bytes: 1000 ou 4000).

        adjacency_matrix (V, V):
            Ganho de canal de interferência entre UEs (escala linear).

        Parameters
        ----------
        state : NetworkState
            Estado completo da rede no TTI atual.

        Returns
        -------
        GraphObservation
            Observação de grafo pronta para consumo.
        """
        cqi = 10 * np.log10(state.direct_gains + 1e-20) + 200

        node_features = np.column_stack([
            cqi,
            state.queues,
            state.cbr_bytes,
        ]).astype(np.float32)

        adjacency_matrix = state.interference_gains.astype(np.float32)

        return GraphObservation(
            node_features=node_features,
            adjacency_matrix=adjacency_matrix,
        )

    def to_dict(self, obs: GraphObservation) -> dict[str, np.ndarray]:
        """Converte GraphObservation para dict Gymnasium-compatível.

        Parameters
        ----------
        obs : GraphObservation
            Observação de grafo.

        Returns
        -------
        dict[str, np.ndarray]
            Dicionário com 'node_features' e 'adjacency_matrix'.
        """
        return {
            "node_features": obs.node_features,
            "adjacency_matrix": obs.adjacency_matrix,
        }
