"""Contrato abstrato para funções de recompensa."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RewardFunction(ABC):
    """Interface para cálculo de recompensa.

    Permite trocar a função de reward (ex: throughput puro,
    fairness de Jain, penalidade de delay) sem alterar o step().
    """

    @abstractmethod
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
            Recompensa escalar.
        """
        ...
