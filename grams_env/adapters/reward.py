"""Função de recompensa: throughput − penalidade de fila."""

from __future__ import annotations

import numpy as np

from grams_env.core.ports.reward_port import RewardFunction


class ThroughputQueueReward(RewardFunction):
    """Recompensa = soma do throughput − peso × soma das filas.

    Incentiva alta taxa de transferência e baixa latência (filas pequenas).
    """

    def __init__(self, weight: float = 1e-4) -> None:
        """
        Parameters
        ----------
        weight : float
            Peso da penalidade de fila (default: 1e-4).
        """
        self._weight = weight

    def compute(
        self,
        real_throughput: np.ndarray,
        queues: np.ndarray,
    ) -> float:
        """Calcula a recompensa escalar do TTI.

        Parameters
        ----------
        real_throughput : np.ndarray
            Throughput real por UE (V,) em bits.
        queues : np.ndarray
            Filas residuais por UE (V,) em bits.

        Returns
        -------
        float
            Recompensa escalar: throughput_total − weight × queue_total.
        """
        total_throughput = float(np.sum(real_throughput))
        queue_penalty = float(np.sum(queues)) * self._weight
        return total_throughput - queue_penalty
