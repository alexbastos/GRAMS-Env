"""Contrato abstrato para modelos de propagação."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PropagationModel(ABC):
    """Interface para cálculo de path loss.

    Qualquer modelo 3GPP (UMa, UMi, InH) pode implementar
    esta interface sem alterar os serviços que a consomem.
    """

    @abstractmethod
    def path_loss_direct_db(
        self,
        d_2d: np.ndarray,
        is_los: np.ndarray,
    ) -> np.ndarray:
        """Calcula o path loss gNB→UE em dB.

        Parameters
        ----------
        d_2d : np.ndarray
            Distância 2D (V,) em metros.
        is_los : np.ndarray
            Estado LOS de cada UE (V,).

        Returns
        -------
        np.ndarray
            Path loss em dB (V,).
        """
        ...

    @abstractmethod
    def path_loss_inter_ue_db(
        self,
        dist_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calcula o path loss entre pares de UEs em dB.

        Parameters
        ----------
        dist_matrix : np.ndarray
            Matriz de distâncias (V, V) em metros.

        Returns
        -------
        np.ndarray
            Path loss em dB (V, V).
        """
        ...

    @abstractmethod
    def los_probability(self, d_2d: np.ndarray) -> np.ndarray:
        """Probabilidade de LOS com base na distância."""
        ...
