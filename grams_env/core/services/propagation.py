"""Modelo de propagação 3GPP TR 36.873 UMa — domínio puro."""

from __future__ import annotations

import numpy as np

from grams_env.core.domain.cell import CellConfig
from grams_env.core.ports.propagation_port import PropagationModel


class TR36873_UMa(PropagationModel):
    """Implementação do path loss UMa conforme 3GPP TR 36.873.

    Recebe CellConfig via injeção de dependência. Sem herança
    de gymnasium, sem estado mutável, 100% testável isoladamente.
    """

    def __init__(self, config: CellConfig) -> None:
        self._cfg = config

    def los_probability(self, d_2d: np.ndarray) -> np.ndarray:
        """Calcula P_LOS(d) = min(18/d, 1)·(1 − exp(−d/63)) + exp(−d/63)."""
        d = np.clip(d_2d, 1.0, None)
        return (
            np.minimum(18.0 / d, 1.0) * (1 - np.exp(-d / 63))
            + np.exp(-d / 63)
        )

    def path_loss_direct_db(
        self,
        d_2d: np.ndarray,
        is_los: np.ndarray,
    ) -> np.ndarray:
        """Calcula o path loss UE–gNB (3GPP TR 36.873, Tab 7.2-1, 3D-UMa).

        LOS:
            PL = 22.0·log₁₀(d_3D) + 28.0 + 20·log₁₀(f_c)

        NLOS:
            PL = 161.04 − 7.1·log₁₀(W) + 7.5·log₁₀(h)
                 − (24.37 − 3.7·(h/h_BS)²)·log₁₀(h_BS)
                 + (43.42 − 3.1·log₁₀(h_BS))·(log₁₀(d_3D) − 3)
                 + 20·log₁₀(f_c)
                 − (3.2·(log₁₀(17.625))² − 4.97)
                 − 0.6·(h_UT − 1.5)

        Parameters
        ----------
        d_2d : np.ndarray
            Distância 2D (V,) de cada UE ao gNB em metros.
        is_los : np.ndarray
            Estado LOS/NLOS de cada UE (V,).

        Returns
        -------
        np.ndarray
            Path loss em dB (V,).
        """
        c = self._cfg
        d_safe = np.clip(d_2d, 10.0, None)
        d_3d = np.sqrt(d_safe**2 + c.delta_h**2)

        # LOS path loss
        pl_los = (
            22.0 * np.log10(d_3d)
            + 28.0
            + 20.0 * np.log10(c.carrier_freq_ghz)
        )

        # NLOS path loss
        pl_nlos = (
            161.04
            - 7.1 * np.log10(c.street_width_m)
            + 7.5 * np.log10(c.building_height_m)
            - (24.37 - 3.7 * (c.building_height_m / c.gnb_height_m) ** 2)
            * np.log10(c.gnb_height_m)
            + (43.42 - 3.1 * np.log10(c.gnb_height_m))
            * (np.log10(d_3d) - 3)
            + 20.0 * np.log10(c.carrier_freq_ghz)
            - (3.2 * (np.log10(17.625)) ** 2 - 4.97)
            - 0.6 * (c.ue_height_m - 1.5)
        )

        # Selecionar com base no estado LOS/NLOS
        return np.where(is_los, pl_los, np.maximum(pl_los, pl_nlos))

    def path_loss_inter_ue_db(
        self, dist_matrix: np.ndarray
    ) -> np.ndarray:
        """Calcula o path loss entre pares de UEs (3GPP UMa NLOS simplificado).

        Como UEs estão à mesma altura, usa-se a fórmula NLOS com d_3D ≈ d_2D.

        Parameters
        ----------
        dist_matrix : np.ndarray
            Matriz (V, V) de distâncias 2D entre UEs.

        Returns
        -------
        np.ndarray
            Path loss em dB (V, V).
        """
        c = self._cfg
        d = np.clip(dist_matrix, 1.0, None)

        return (
            161.04
            - 7.1 * np.log10(c.street_width_m)
            + 7.5 * np.log10(c.building_height_m)
            - (24.37 - 3.7 * (c.building_height_m / c.gnb_height_m) ** 2)
            * np.log10(c.gnb_height_m)
            + (43.42 - 3.1 * np.log10(c.gnb_height_m))
            * (np.log10(d) - 3)
            + 20.0 * np.log10(c.carrier_freq_ghz)
            - (3.2 * (np.log10(17.625)) ** 2 - 4.97)
            - 0.6 * (c.ue_height_m - 1.5)
        )
